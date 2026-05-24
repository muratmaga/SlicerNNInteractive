import io
import gzip
import logging
import requests
import copy
import subprocess
import sys
import threading
import time

import importlib.util

import numpy as np
from pathlib import Path

import slicer
import qt
import vtk
from qt import QApplication, QPalette

from vtkmodules.util.numpy_support import vtk_to_numpy

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox


###############################################################################
# Constants for large-volume dynamic-ROI mode
###############################################################################
# Total upload voxel budget: a 512^3 ≈ 134M-voxel cube. This is the GPU VRAM
# budget — at 4 bytes per float32 voxel plus the model's autozoom replicas,
# this keeps the server inside ~16 GB even on modest GPUs. The crop window is
# allocated against this budget anisotropically, matching the data's shape.
ROI_BUDGET_VOXEL_CUBE = 512
# If True, always use ROI mode regardless of the size heuristic. For testing.
FORCE_ROI_MODE = False
# Auto-enable ROI mode when the volume exceeds this many voxels (I*J*K).
# The threshold is comfortably above the budget so volumes that fit natively
# don't get cropped unnecessarily.
ROI_AUTO_ENABLE_VOXEL_THRESHOLD = 1_000_000_000
# Reference physical voxel spacing the nnInteractive checkpoint was trained on.
# Used to compute max_zoom_out_factor from the data's spacing.
ROI_REFERENCE_SPACING_MM = 1.0
# Upper bound for the auto-computed max_zoom_out_factor.
ROI_MAX_ZOOM_OUT_FACTOR_CAP = 64.0
# Fraction of the crop window that counts as the "comfort zone". A new prompt
# inside the central ROI_COMFORT_FRACTION × window reuses the existing crop;
# otherwise the crop window recenters and re-uploads. 2/3 gives ~1/6 of the
# window on each side as headroom for the model to expand the segmentation.
ROI_COMFORT_FRACTION = 2.0 / 3.0


###############################################################################
# SlicerNNInteractive
###############################################################################


class SlicerNNInteractive(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("nnInteractive")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Segmentation")
        ]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente", "Andras Lasso", "Kiran Vaidhya Venkadesh", "Bram van Ginneken", "Clara I. Sanchez"]
        self.parent.helpText = """
            This is an 3D Slicer extension for using nnInteractive.

            Read more about this plugin here: https://github.com/coendevente/SlicerNNInteractive.
            """
        self.parent.acknowledgementText = """When using SlicerNNInteractive, please cite as described here: https://github.com/coendevente/SlicerNNInteractive?tab=readme-ov-file#citation."""


###############################################################################
# SlicerNNInteractiveWidget
###############################################################################


class SlicerNNInteractiveWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    INTERNAL_SERVER_URL = "http://127.0.0.1:1527"

    def ensure_synched(func):
        """
        Decorator that ensures the image and segment are synced before calling
        the actual prompt function.
        """
        def inner(self, *args, **kwargs):
            self.install_dependencies()

            if self._internal_server_mode and not self._server_launching_dependencies_installed:
                return

            if self._internal_server_mode and not self._is_internal_server_running():
                started = self.start_internal_server()
                if started:
                    progressbar = slicer.util.createProgressDialog(autoClose=False)
                    progressbar.minimum = 0
                    progressbar.maximum = 0
                    progressbar.setLabelText("Waiting for nnInteractive server to start...")
                    slicer.app.processEvents()
                    ready = self._wait_for_server_ready(timeout=120)
                    progressbar.close()
                    if not ready:
                        error_detail = getattr(self, "_server_last_error", "").strip()
                        if error_detail:
                            msg = f"nnInteractive server failed to start.\n\nServer output:\n{error_detail}"
                        else:
                            msg = (
                                "nnInteractive server did not start in time. "
                                "Please try again or start it manually using the 'Start Server' button."
                            )
                        slicer.util.errorDisplay(msg)
                        return

            failed_to_sync = False
            image_was_uploaded = False

            if self.image_changed():
                logging.debug("Image changed (or not previously set). Calling upload_image_to_server()")
                if self._roi_active:
                    slicer.util.showStatusMessage(
                        f"nnInteractive: cropping window {self._roi_shape_ijk} and uploading...", 0)
                else:
                    slicer.util.showStatusMessage("nnInteractive: uploading full volume...", 0)
                slicer.app.processEvents()
                result = self.upload_image_to_server()
                failed_to_sync = result is None
                image_was_uploaded = not failed_to_sync

            # Image upload resets server-side interactions, so the segment must
            # follow the image whenever the image was just (re)uploaded. In ROI
            # mode this also ensures the freshly cropped segment is in sync.
            if not failed_to_sync and (image_was_uploaded or self.selected_segment_changed()):
                logging.debug("Calling upload_segment_to_server() (image_was_uploaded=%s)", image_was_uploaded)
                slicer.util.showStatusMessage("nnInteractive: uploading segment...", 0)
                slicer.app.processEvents()
                self.remove_all_but_last_prompt()
                result = self.upload_segment_to_server()
                failed_to_sync = result is None
            else:
                logging.debug("Segment did not change!")

            if not failed_to_sync:
                slicer.util.showStatusMessage("nnInteractive: running inference...", 0)
                slicer.app.processEvents()
                try:
                    return func(self, *args, **kwargs)
                finally:
                    slicer.util.showStatusMessage("", 0)

        return inner

    ###############################################################################
    # Setup and initialization functions
    ###############################################################################

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self._server_connection_dependencies_installed = False
        self._server_launching_dependencies_installed = False
        self._server_process = None
        self._internal_server_mode = True
        self._server_log_lock = threading.Lock()
        self._server_log_buffer = []
        self._server_log_threads = []
        self._server_log_timer = None

        # Large-volume dynamic-ROI state.
        # When _roi_active, prompts and image/segment uploads are restricted to
        # a native-resolution crop window around each prompt; server responses
        # are pasted back into the full-volume segment at the same crop offset.
        # No downsampling — the window shape is sized against ROI_BUDGET_VOXEL_CUBE^3
        # to preserve the data's aspect ratio.
        self._roi_active = False
        self._roi_start_ijk = None       # (i, j, k) start in original volume IJK
        self._roi_shape_ijk = None       # (di, dj, dk) extent in original volume IJK
        # User override for max_zoom_out_factor; None means auto-compute from spacing.
        self._roi_max_zoom_out_factor_override = None

    def setup(self):
        """
        Overridden setup method. Initializes UI and setups up prompts.
        """
        ScriptedLoadableModuleWidget.setup(self)

        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SlicerNNInteractive.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.scribble_segment_node_name = "ScribbleSegmentNode (do not touch)"

        # Set up editor widget
        self.ui.editor_widget.setMaximumNumberOfUndoStates(10)
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        # Use the same segmentation parameter node as the Segment Editor core module
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode")
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        self.ui.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.ui.editor_widget.setSegmentationNode(self.get_segmentation_node())

        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "button": self.ui.pbInteractionPoint,
                "button_text": self.ui.pbInteractionPoint.text,
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
                "button_icon_filename": "bbox_icon.svg",
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "button": self.ui.pbInteractionLasso,
                "button_text": self.ui.pbInteractionLasso.text,
                "button_icon_filename": "lasso_icon.svg",
            },
        }

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        self.setup_prompts()

        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """

        # On macOS, internal server is not supported; force external mode
        is_macos = sys.platform == "darwin"
        if is_macos:
            self._internal_server_mode = False
            self.ui.rbInternalServer.setEnabled(False)
            self.ui.rbInternalServer.setToolTip(
                "<html>Internal server is not available on macOS. "
                "An external server must be set up - see "
                "<a href='https://github.com/coendevente/SlicerNNInteractive#server-side'>server-side setup instructions</a>.</html>"
            )
            self.ui.rbExternalServer.setChecked(True)
            self.ui.internalServerWidget.setEnabled(False)
            self.ui.externalServerWidget.setEnabled(True)
        else:
            saved_mode = slicer.util.settingsValue("SlicerNNInteractive/serverMode", "internal")
            self._internal_server_mode = (saved_mode == "internal")
            self.ui.rbInternalServer.setChecked(self._internal_server_mode)
            self.ui.rbExternalServer.setChecked(not self._internal_server_mode)
            self.ui.internalServerWidget.setEnabled(self._internal_server_mode)
            self.ui.externalServerWidget.setEnabled(not self._internal_server_mode)
            self.ui.rbInternalServer.toggled.connect(self.on_server_mode_changed)
        self.ui.pbStartStopServer.clicked.connect(self.on_start_stop_server_clicked)
        self.update_start_stop_button()

        # Load saved external server URL
        savedServer = slicer.util.settingsValue("SlicerNNInteractive/server", "http://localhost:1527")
        self.ui.Server.text = savedServer
        self.server = self.INTERNAL_SERVER_URL if self._internal_server_mode else savedServer.rstrip("/")

        self.ui.Server.editingFinished.connect(self.update_server)
        self.ui.pbTestServer.clicked.connect(self.test_server_connection)

        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)

        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(
            self.on_prompt_type_positive_clicked
        )
        self.ui.pbPromptTypeNegative.clicked.connect(
            self.on_prompt_type_negative_clicked
        )

        self.ui.pbInteractionLassoCancel.setVisible(False)
        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)

        self.ui.pbInteractionLassoCancel.clicked.connect(self.on_lasso_cancel_clicked)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)

        self.ui.pbClearServerOutput.clicked.connect(
            lambda: self.ui.serverOutputTextEdit.clear()
        )
        self._init_server_log_polling()

        self._build_roi_mode_ui()

    def _build_roi_mode_ui(self):
        """Insert a 'Large-volume mode' groupbox into the Configuration tab,
        between Server Settings and Server output. Shows clearly whether
        large-volume mode is on/off for the next prompt, and lets the user
        override the auto-computed max_zoom_out_factor."""
        try:
            config_layout = self.ui.serverGroup.parent().layout()
            if config_layout is None:
                return

            group = qt.QGroupBox("Large-volume mode")
            vbox = qt.QVBoxLayout(group)

            self.ui_roiModeBadge = qt.QLabel("MODE: …")
            badge_font = qt.QFont()
            badge_font.setBold(True)
            badge_font.setPointSize(13)
            self.ui_roiModeBadge.setFont(badge_font)
            vbox.addWidget(self.ui_roiModeBadge)

            self.ui_roiReasonLabel = qt.QLabel("")
            self.ui_roiReasonLabel.setWordWrap(True)
            vbox.addWidget(self.ui_roiReasonLabel)

            self.ui_roiDetailsLabel = qt.QLabel("")
            self.ui_roiDetailsLabel.setWordWrap(True)
            self.ui_roiDetailsLabel.setStyleSheet("color: #555; font-family: monospace;")
            vbox.addWidget(self.ui_roiDetailsLabel)

            row1 = qt.QHBoxLayout()
            row1.addWidget(qt.QLabel("Max zoom-out factor:"))
            self.ui_zoomFactorSpin = qt.QDoubleSpinBox()
            self.ui_zoomFactorSpin.setDecimals(1)
            self.ui_zoomFactorSpin.setSingleStep(1.0)
            self.ui_zoomFactorSpin.setRange(0.0, ROI_MAX_ZOOM_OUT_FACTOR_CAP)
            self.ui_zoomFactorSpin.setValue(0.0)  # 0 = auto
            self.ui_zoomFactorSpin.setToolTip(
                "0 = auto (computed from voxel spacing). "
                "Set to a positive number to override."
            )
            self.ui_zoomFactorSpin.valueChanged.connect(self._on_zoom_factor_override_changed)
            row1.addWidget(self.ui_zoomFactorSpin)
            self.ui_zoomFactorAutoLabel = qt.QLabel("")
            row1.addWidget(self.ui_zoomFactorAutoLabel)
            row1.addStretch(1)
            vbox.addLayout(row1)

            self.ui_roiRefreshButton = qt.QPushButton("Refresh from current volume")
            self.ui_roiRefreshButton.setToolTip(
                "Recompute the mode badge from the currently selected volume."
            )
            self.ui_roiRefreshButton.clicked.connect(self._refresh_roi_status_label)
            vbox.addWidget(self.ui_roiRefreshButton)

            # Insert just before the Server output group
            insert_at = config_layout.count()
            for i in range(config_layout.count()):
                w = config_layout.itemAt(i).widget()
                if w is not None and w.objectName == "serverOutputGroup":
                    insert_at = i
                    break
            config_layout.insertWidget(insert_at, group)

            self._refresh_roi_status_label()
        except Exception as e:
            logging.debug(f"Failed to build ROI mode UI: {e}")

    def _on_zoom_factor_override_changed(self, value):
        if value <= 0.0:
            self._roi_max_zoom_out_factor_override = None
        else:
            self._roi_max_zoom_out_factor_override = float(value)
        self._refresh_roi_status_label()

    def _refresh_roi_status_label(self):
        """Compute and display the large-volume mode status. Reflects the
        active ROI when one has been established, otherwise previews what
        will happen on the next prompt based on the current volume."""
        if not hasattr(self, "ui_roiModeBadge"):
            return

        spacing = self._volume_spacing_mm()
        vol_ijk = self._volume_shape_ijk()
        will_be_on = self._should_use_roi_mode()

        # Badge
        if will_be_on:
            self.ui_roiModeBadge.setText("MODE: ON  (large-volume / ROI)")
            self.ui_roiModeBadge.setStyleSheet(
                "color: white; background-color: #2e8b57; padding: 4px 8px; border-radius: 3px;"
            )
        else:
            self.ui_roiModeBadge.setText("MODE: OFF  (full volume sent as-is)")
            self.ui_roiModeBadge.setStyleSheet(
                "color: white; background-color: #888; padding: 4px 8px; border-radius: 3px;"
            )

        # Reason — short, factual, voxel-count based.
        if vol_ijk is None:
            self.ui_roiReasonLabel.setText("No volume selected yet.")
        else:
            total = int(vol_ijk[0]) * int(vol_ijk[1]) * int(vol_ijk[2])
            gv = total / 1e9
            thresh_gv = ROI_AUTO_ENABLE_VOXEL_THRESHOLD / 1e9
            if FORCE_ROI_MODE:
                self.ui_roiReasonLabel.setText(
                    f"FORCE_ROI_MODE is True (override). Volume size: {gv:.3f} GV."
                )
            elif will_be_on:
                self.ui_roiReasonLabel.setText(
                    f"Volume size: {gv:.3f} GV  (> {thresh_gv:g} GV threshold)."
                )
            else:
                self.ui_roiReasonLabel.setText(
                    f"Volume size: {gv:.3f} GV  (≤ {thresh_gv:g} GV threshold)."
                )

        # Details: actual crop window if active, otherwise preview of what
        # would happen. Native resolution throughout — no downsampling.
        if self._roi_active and self._roi_start_ijk is not None:
            mzof = self._compute_max_zoom_out_factor()
            self.ui_roiDetailsLabel.setText(
                f"Crop window active:  start_ijk={self._roi_start_ijk}\n"
                f"                     shape_ijk={self._roi_shape_ijk}  (native res)\n"
                f"                     zoom-out={mzof}"
            )
        elif will_be_on and vol_ijk is not None:
            shape = self._compute_window_shape()
            mzof = self._compute_max_zoom_out_factor()
            if shape is not None:
                self.ui_roiDetailsLabel.setText(
                    f"Preview (no prompt yet):  window_ijk≈{shape}  (native res)\n"
                    f"                          zoom-out≈{mzof}"
                )
            else:
                self.ui_roiDetailsLabel.setText("")
        elif vol_ijk is not None:
            self.ui_roiDetailsLabel.setText(
                f"Volume shape (I,J,K) = {vol_ijk}, spacing = "
                f"{tuple(round(s, 4) for s in spacing) if spacing else '?'} mm"
            )
        else:
            self.ui_roiDetailsLabel.setText("")

        # Auto-zoom-factor hint
        if hasattr(self, "ui_zoomFactorAutoLabel"):
            mzof = self._compute_max_zoom_out_factor()
            if self._roi_max_zoom_out_factor_override is None and mzof is not None:
                self.ui_zoomFactorAutoLabel.setText(f"(auto = {mzof})")
            else:
                self.ui_zoomFactorAutoLabel.setText("(override)")

    def on_server_mode_changed(self, internal_selected):

        # On macOS, internal server is not supported.
        # Do not save the current choice in settings, as users actually prefer an external server,
        # there is just no other option for now.
        is_macos = sys.platform == "darwin"
        if is_macos:
            return

        self._internal_server_mode = internal_selected
        settings = qt.QSettings()
        settings.setValue("SlicerNNInteractive/serverMode", "internal" if internal_selected else "external")
        self.ui.internalServerWidget.setEnabled(internal_selected)
        self.ui.externalServerWidget.setEnabled(not internal_selected)
        if internal_selected:
            self.server = self.INTERNAL_SERVER_URL
        else:
            self.stop_internal_server()
            self.server = self.ui.Server.text.rstrip("/")

    def on_start_stop_server_clicked(self):
        if self._is_internal_server_running():
            self.stop_internal_server()
        else:
            self.start_internal_server()

    def start_internal_server(self):
        if self._is_internal_server_running():
            return True
        # Installed/built layout: nninteractive_slicer_server/ sits next to this script.
        # Source-tree layout (development): server/ is two directories above slicer_plugin/.
        server_main = Path(__file__).parent / "nninteractive_slicer_server" / "main.py"
        if not server_main.exists():
            server_main = Path(__file__).parents[2] / "server" / "nninteractive_slicer_server" / "main.py"
        if not server_main.exists():
            logging.error(f"Server script not found: {server_main}")
            self.update_start_stop_button()
            return False
        server_cmd = [sys.executable, str(server_main)]
        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self._server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **kwargs,
            )
            logging.info(f"Started nnInteractive server (PID: {self._server_process.pid})")
            self._start_server_log_pump()
            self.update_start_stop_button()
            return True
        except Exception as e:
            logging.error(f"Failed to start internal server: {e}")
            self._server_process = None
            self.update_start_stop_button()
            return False

    def stop_internal_server(self):
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None
        try:
            self.update_start_stop_button()
        except Exception:
            pass

    def _append_server_log(self, text):
        if not text:
            return
        with self._server_log_lock:
            self._server_log_buffer.append(text)

    def _consume_server_logs(self):
        with self._server_log_lock:
            if not self._server_log_buffer:
                return ""
            combined = "\n".join(self._server_log_buffer)
            self._server_log_buffer = []
            return combined

    def _read_server_stream(self, stream):
        try:
            while True:
                raw_line = stream.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line:
                    self._append_server_log(line)
        except Exception as e:
            self._append_server_log(f"[log reader error: {e}]")

    def _start_server_log_pump(self):
        self._server_log_threads = []
        if self._server_process is None:
            return
        for stream in [self._server_process.stdout, self._server_process.stderr]:
            if stream is None:
                continue
            t = threading.Thread(
                target=self._read_server_stream,
                args=(stream,),
                daemon=True,
            )
            t.start()
            self._server_log_threads.append(t)

    def _init_server_log_polling(self):
        if self._server_log_timer is not None:
            return
        self._server_log_timer = qt.QTimer()
        self._server_log_timer.setInterval(250)
        self._server_log_timer.connect("timeout()", self._poll_server_logs)
        self._server_log_timer.start()

    def _poll_server_logs(self):
        text = self._consume_server_logs()
        if not text:
            return
        try:
            self.ui.serverOutputTextEdit.appendPlainText(text)
        except Exception:
            pass

    def _is_internal_server_running(self):
        return self._server_process is not None and self._server_process.poll() is None

    def update_start_stop_button(self):
        try:
            if self._is_internal_server_running():
                self.ui.pbStartStopServer.setText("Stop Server")
            else:
                self.ui.pbStartStopServer.setText("Start Server")
        except Exception:
            pass

    def _wait_for_server_ready(self, timeout=120):
        self._server_last_error = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._is_internal_server_running():
                # Process died - let reader threads drain, then capture output
                for t in self._server_log_threads:
                    t.join(timeout=1.0)
                self._server_last_error = self._consume_server_logs()
                self._server_process = None
                self.update_start_stop_button()
                return False
            try:
                requests.get(self.server, timeout=1)
                self.update_start_stop_button()
                return True
            except Exception:
                pass
            slicer.app.processEvents()
            time.sleep(0.5)
        return False

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "Shift+L": self.submit_lasso_if_present,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
        }
        self.shortcut_items = {}

        for shortcut_key, shortcut_event in shortcuts.items():
            logging.debug(f"Added shortcut for {shortcut_key}: {shortcut_event}")
            shortcut = qt.QShortcut(
                qt.QKeySequence(shortcut_key), slicer.util.mainWindow()
            )
            shortcut.activated.connect(shortcut_event)
            self.shortcut_items[shortcut_key] = shortcut

    def remove_shortcut_items(self):
        """
        Called at cleanup to remove all the shortcuts we attached.
        """
        if hasattr(self, "shortcut_items"):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()
                shortcut = None

    def install_dependencies(self):
        """
        Installs Python packages needed by the module.
        Connection dependencies (requests_toolbelt, scikit-image) are always installed.
        Server-launching dependencies (NNUNet, nnInteractive, server runtime) are only
        installed when the internal server mode is active.
        """
        if not self._server_connection_dependencies_installed:
            self._install_server_connection_dependencies()

        if self._internal_server_mode and not self._server_launching_dependencies_installed:
            self._install_server_launching_dependencies()

    def _install_server_connection_dependencies(self):
        deps = [
            ("requests_toolbelt", "requests_toolbelt"),
            ("skimage", "scikit-image"),
        ]
        for import_name, pkg in deps:
            if not self.check_dependency_installed(import_name, pkg):
                slicer.util.pip_install(pkg)
        self._server_connection_dependencies_installed = True

    def _install_server_launching_dependencies(self):
        if not self.isNNUNetModuleInstalled():
            raise RuntimeError(
                "The internal server requires the NNUNet extension."
                " Please install the NNUNet extension and restart to proceed."
            )

        if not self._installNNUNetIfNeeded():
            raise RuntimeError("The internal server requires the NNUNet Python package.")

        deps = [
            ("requests_toolbelt", "requests_toolbelt"),
            ("skimage", "scikit-image"),
            # Dependencies for the local nninteractive server (server/nninteractive_slicer_server/main.py).
            # nnInteractive>=1.1.5 requires nnunetv2>=2.7.0 (compatible with the installed version).
            ("nnInteractive", "nnInteractive>=1.1.5"),
            ("uvicorn", "uvicorn"),
            ("xxhash", "xxhash"),
            ("fastapi", "fastapi"),
            ("multipart", "python-multipart"),
            ("huggingface_hub", "huggingface_hub"),
        ]
        for import_name, pkg in deps:
            if not self.check_dependency_installed(import_name, pkg):
                slicer.util.pip_install(pkg)
        self._server_launching_dependencies_installed = True
        # Internal server launching dependencies include server connection dependencies
        self._server_connection_dependencies_installed = True

    @staticmethod
    def isNNUNetModuleInstalled():
        try:
            import SlicerNNUNetLib
            return True
        except ImportError:
            return False

    def _installNNUNetIfNeeded(self) -> bool:
        from SlicerNNUNetLib import InstallLogic
        logic = InstallLogic()
        return logic.setupPythonRequirements()

    def check_dependency_installed(self, import_name, module_name_and_version):
        """
        Checks if a package is importable and satisfies the version requirement.
        Accepts any PEP 440 specifier (e.g. 'pkg==1.2', 'pkg>=1.1.5', 'pkg').
        """
        from packaging.requirements import Requirement
        from packaging.version import Version
        import importlib.metadata as metadata

        req = Requirement(module_name_and_version)

        if importlib.util.find_spec(import_name) is None:
            return False

        if req.specifier:
            try:
                installed = Version(metadata.version(req.name))
                if installed not in req.specifier:
                    return False
            except metadata.PackageNotFoundError:
                logging.debug(f"Could not determine version for {req.name}.")

        return True

    def run_with_progress_bar(self, target, args, title):
        """
        Runs a function in a background thread, while showing a progress bar in the UI
        as a pop up window.
        """
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)

        parallel_event = threading.Event()
        dep_thread = threading.Thread(
            target=target,
            args=(
                *args,
                parallel_event,
            ),
        )
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()

    def cleanup(self):
        """
        Clean up resources when the module is closed.
        """
        self.removeObservers()

        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()
        if self._server_log_timer is not None:
            self._server_log_timer.stop()
            self._server_log_timer = None
        self.stop_internal_server()

    def __del__(self):
        """
        Called when the widget is destroyed.
        """
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()

        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(
                prompt_type["name"]
            ):
                logging.debug("Skipping %s", prompt_name)
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()

            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)

            prompt_type["button"].setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )

            self.prev_caller = None

            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    prompt_type["on_placed_function"],
                )

            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name)) 
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/{prompt_type['button_icon_filename']}"))
            prompt_type["button"].setIcon(icon)

        if (
            not skip_if_exists
            or slicer.mrmlScene.GetFirstNodeByName(self.scribble_segment_node_name)
            is None
        ):
            self.setup_scribble_prompt()

            self.ui.pbInteractionScribble.setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )
            self.all_prompt_buttons["scribble"] = self.ui.pbInteractionScribble

        # To make sure that when segment is reset, no interaction is selected (without this code
        # the last interaction tool gets selected)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

    def setup_scribble_prompt(self):
        """
        Creates a hidden "Segment Editor" for the scribble prompt.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt

        # Create a background (headless) segment editor
        self.scribble_editor_widget = (
            qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        )
        self.scribble_editor_widget.setMRMLScene(slicer.mrmlScene)
        self.scribble_editor_widget.setMaximumNumberOfUndoStates(10)

        # Create a separate SegmentEditorNode
        self.scribble_editor_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode"
        )
        self.scribble_editor_widget.setMRMLSegmentEditorNode(self.scribble_editor_node)

        self.scribble_segment_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        self.scribble_segment_node.SetReferenceImageGeometryParameterFromVolumeNode(
            self.get_volume_node()
        )
        self.scribble_segment_node.SetName(self.scribble_segment_node_name)

        # Make sure the node exists and is set
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)

        self.scribble_segment_node.CreateDefaultDisplayNodes()
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "bg", "bg", [0.0, 0.0, 1.0]
        )
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "fg", "fg", [0.0, 0.0, 1.0]
        )
        dn = self.scribble_segment_node.GetDisplayNode()

        opacity = 0.2
        dn.SetSegmentOpacity2DFill("bg", opacity)
        dn.SetSegmentOpacity2DOutline("bg", opacity)
        dn.SetSegmentOpacity2DFill("fg", opacity)
        dn.SetSegmentOpacity2DOutline("fg", opacity)

        self._prev_scribble_mask = None
            
        light_dark_mode = self.is_ui_dark_or_light_mode()
        icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/scribble_icon.svg"))
        self.ui.pbInteractionScribble.setIcon(icon)

    def is_ui_dark_or_light_mode(self):
        # Returns whether the current appearance of the UI is dark mode (will return "dark")
        # or light mode (will return "light")
        current_style = slicer.app.settings().value("Styles/Style")

        if current_style == "Dark Slicer":
            return "dark"
        elif current_style == "Light Slicer":
            return "light"
        elif current_style == "Slicer":
            app_palette = QApplication.instance().palette()
            window_color = app_palette.color(QPalette.Active, QPalette.Window)
            lightness = window_color.lightness()
            dark_mode_threshold = 128

            if lightness < dark_mode_threshold:
                return "dark"
            else:
                return "light"
        return "light"

    def remove_prompt_nodes(self):
        """
        Removes all the Markups/Fiducials prompts.
        """

        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)

        for prompt_type in list(self.prompt_types.values()):
            _remove(prompt_type["name"])

        self.ui.pbInteractionLassoCancel.setVisible(False)

        _remove(self.scribble_segment_node_name)

    def on_interaction_node_modified(self, caller, event):
        """
        Deselect prompt button if interaction mode is not place point anymore
        """

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        for prompt_type in self.prompt_types.values():
            if interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place:
                if prompt_type["name"] == "LassoPrompt" and (self.ui.pbInteractionLasso.isChecked()):
                    self.submit_lasso_if_present()
                prompt_type["button"].setChecked(False)
            elif interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
                placingThisNode = (selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID())
                prompt_type["button"].setChecked(placingThisNode)

        # Stop scribble if placing markup
        if interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
            self.ui.pbInteractionScribble.setChecked(False)

    def remove_all_but_last_prompt(self):
        """
        Removes all but the most recently placed markup points
        (helpful when segment change was detected).
        """
        last_modified_node = None
        all_nodes = []

        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)

                    all_nodes.append(node)
                    if (
                        last_modified_node is None
                        or node.GetMTime() > last_modified_node.GetMTime()
                    ):
                        last_modified_node = node

        for node in all_nodes:
            n = node.GetNumberOfControlPoints()

            if node == last_modified_node:
                if node.GetName() == "LassoPrompt":
                    continue
                n -= 1

            for i in range(n):
                node.RemoveNthControlPoint(0)

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if checked:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(self.prompt_types[prompt_name]["node_class"])
            selectionNode.SetActivePlaceNodeID(self.prompt_types[prompt_name]["node"].GetID())
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            if prompt_name == "lasso":
                self.submit_lasso_if_present()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def display_node_markup_point(self, display_node):
        """
        Handles the appearance of the point display node.
        """
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(0.0, 0.0, 1.0)  # Green color
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views

    def display_node_markup_bbox(self, display_node):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    def display_node_markup_lasso(self, display_node):
        """
        Handles the appearance of the lasso display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetGlyphScale(1)
        display_node.SetLineThickness(0.3)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    ###############################################################################
    # Event handlers for prompts
    ###############################################################################

    #
    #  -- Point
    #
    def on_point_placed(self, caller, event):
        """
        Called when a point is placed in the scene. Grabs the point position
        and sends it to the server.
        """
        xyz = self.xyz_from_caller(caller)

        volume_node = self.get_volume_node()
        if volume_node:
            with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):
                self.point_prompt(xyz=xyz, positive_click=self.is_positive)

    def point_prompt(self, xyz=None, positive_click=False):
        """Public entry: decide ROI placement, then send the prompt."""
        if xyz is not None:
            self._maybe_update_roi_for_prompt(tuple(xyz))
        self._send_point_prompt(xyz=xyz, positive_click=positive_click)

    @ensure_synched
    def _send_point_prompt(self, xyz=None, positive_click=False):
        """Uploads point prompt to the server. ROI must already be set up."""
        url = f"{self.server}/add_point_interaction"

        send_xyz = self._ijk_to_roi_local(xyz) if self._roi_active else tuple(xyz)
        seg_response = self.request_to_server(
            url, json={"voxel_coord": list(send_xyz)[::-1], "positive_click": positive_click}
        )

        expected_shape = self._expected_server_shape()
        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False, vol_shape=expected_shape
        )
        logging.debug(f"unpacked_segmentation.sum(): {unpacked_segmentation.sum()}")
        logging.debug(seg_response)
        logging.debug(f"{positive_click} point prompt triggered! {xyz}")

        self.show_segmentation(unpacked_segmentation)

    #
    #  -- Bounding Box
    #
    def on_bbox_placed(self, caller, event):
        """
        Every time a control point is placed/moved for the bounding box ROI node.
        Once two corners are placed, we send the bounding box to the server.
        """
        xyz = self.xyz_from_caller(caller)

        if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
            roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0
            roi_node.SetSize(current_size)

            volume_node = self.get_volume_node()
            if volume_node:
                outer_point_two = self.prev_bbox_xyz

                outer_point_one = [
                    xyz[0] * 2 - outer_point_two[0],
                    xyz[1] * 2 - outer_point_two[1],
                    xyz[2] * 2 - outer_point_two[2],
                ]

                with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):
                    self.bbox_prompt(
                        outer_point_one=outer_point_one,
                        outer_point_two=outer_point_two,
                        positive_click=self.is_positive,
                    )

                def _next():
                    self.setup_prompts()
                    # Start placing a new box
                    self.ui.pbInteractionBBox.click()

                qt.QTimer.singleShot(0, _next)

            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz

        self.prev_caller = caller

    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """Public entry: anchor ROI on bbox center, then send."""
        center = tuple((outer_point_one[a] + outer_point_two[a]) // 2 for a in range(3))
        self._maybe_update_roi_for_prompt(center)
        self._send_bbox_prompt(
            outer_point_one=outer_point_one,
            outer_point_two=outer_point_two,
            positive_click=positive_click,
        )

    @ensure_synched
    def _send_bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """Uploads BBox prompt. ROI-spanning bboxes are silently clipped to fit."""
        url = f"{self.server}/add_bbox_interaction"

        if self._roi_active:
            p1 = self._ijk_to_roi_local(outer_point_one)
            p2 = self._ijk_to_roi_local(outer_point_two)
            roi_shape_kji = self._expected_server_shape()  # native (K, J, I)
            if roi_shape_kji is not None:
                # Clip both points to the ROI-local native extent (IJK order)
                max_ijk = tuple(roi_shape_kji[::-1])  # (I, J, K)
                p1 = tuple(max(0, min(p1[a], max_ijk[a] - 1)) for a in range(3))
                p2 = tuple(max(0, min(p2[a], max_ijk[a] - 1)) for a in range(3))
        else:
            p1 = tuple(outer_point_one)
            p2 = tuple(outer_point_two)

        seg_response = self.request_to_server(
            url,
            json={
                "outer_point_one": list(p1)[::-1],
                "outer_point_two": list(p2)[::-1],
                "positive_click": positive_click,
            },
        )

        expected_shape = self._expected_server_shape()
        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False, vol_shape=expected_shape
        )
        self.show_segmentation(unpacked_segmentation)

    #
    #  -- Lasso
    #
    def on_lasso_placed(self, caller, event):
        """
        Called whenever a new point is added to the lasso.
        """
        pointsDefined = self.prompt_types["lasso"]["node"].GetNumberOfControlPoints() > 0
        self.ui.pbInteractionLassoCancel.setVisible(pointsDefined)

    def on_lasso_cancel_clicked(self):
        """
        Called when the user clicks the cancel button for the lasso.
        """
        self.prompt_types["lasso"]["node"].RemoveAllControlPoints()
        self.ui.pbInteractionLassoCancel.setVisible(False)

    def submit_lasso_if_present(self):
        """
        Submits the currently open lasso. We gather all the control points,
        rasterize them into a mask, and send the mask to the server.
        """
        caller = self.prompt_types["lasso"]["node"]
        xyzs = self.xyz_from_caller(caller, point_type="curve_point")

        if len(xyzs) < 3:
            return

        mask = self.lasso_points_to_mask(xyzs)

        volume_node = self.get_volume_node()
        if volume_node:
            with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):
                self.lasso_or_scribble_prompt(
                    mask=mask, positive_click=self.is_positive, tp="lasso"
                )

            def _next():
                self.setup_prompts()
                # Start placing a new lasso
                self.ui.pbInteractionLasso.click()

            qt.QTimer.singleShot(0, _next)

    #
    #  -- Scribble
    #
    def on_scribble_clicked(self, checked=False):
        """
        Activates/deactivates the hidden Segment Editor's Paint effect on the
        scribble segment (bg or fg, depending on prompt type).
        """
        self.setup_prompts(skip_if_exists=True)

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        if not checked:
            # Deactivate paint effect
            if self.scribble_editor_widget:
                self.scribble_editor_widget.setActiveEffectByName(
                    ""
                )  # Clears the active effect

            # Optionally clear or reset the segmentation node
            if hasattr(self, "_scribble_labelmap_callback_tag"):
                tag = self._scribble_labelmap_callback_tag.get("tag", None)
                if tag:
                    self.scribble_segment_node.RemoveObserver(tag)
                del self._scribble_labelmap_callback_tag

            return

        segment_id = "fg" if self.is_positive else "bg"

        # Set segmentation and segment
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)
        self.scribble_editor_node.SetSelectedSegmentID(segment_id)

        # Set reference volume
        volume_node = self.get_volume_node()
        self.scribble_editor_widget.setSourceVolumeNode(volume_node)

        # Activate paint effect
        self.scribble_editor_widget.setActiveEffectByName("Paint")
        self.scribble_editor_widget.updateWidgetFromMRML()

        paint_effect = self.scribble_editor_widget.activeEffect()
        if paint_effect:
            paint_effect.setParameter("BrushUseAbsoluteSize", "0")  # Use relative mode
            paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            paint_effect.setParameter("BrushRelativeDiameter", ".75")
            self._scribble_labelmap_callback_tag = {
                "tag": self.scribble_segment_node.AddObserver(
                    vtk.vtkCommand.AnyEvent, self.on_scribble_finished
                ),
                "label_name": segment_id,
            }
        logging.debug(f"Scribble mode (hidden editor) activated on '{segment_id}'")

    #
    #  -- Lasso/scribble
    #
    def lasso_or_scribble_prompt(self, mask, positive_click=False, tp="lasso"):
        """Public entry: anchor ROI on mask bbox center, then send.
        Bbox center via axis projections is much cheaper than np.argwhere().mean()
        on a multi-GB mask — no per-voxel coord array allocation, and the per-axis
        np.any() reductions short-circuit per voxel."""
        center_ijk = self._mask_bbox_center_ijk(mask)
        if center_ijk is None:
            return  # mask is empty
        self._maybe_update_roi_for_prompt(center_ijk)
        self._send_lasso_or_scribble_prompt(mask=mask, positive_click=positive_click, tp=tp)

    @staticmethod
    def _mask_bbox_center_ijk(mask):
        """Returns (i, j, k) center of the mask's non-zero bounding box,
        or None if the mask is empty. Mask is (K, J, I) numpy."""
        centers_kji = []
        for axis in range(3):
            other_axes = tuple(a for a in range(3) if a != axis)
            proj = mask.any(axis=other_axes)
            nz = np.where(proj)[0]
            if len(nz) == 0:
                return None
            centers_kji.append(int((int(nz[0]) + int(nz[-1])) // 2))
        return (centers_kji[2], centers_kji[1], centers_kji[0])  # KJI -> IJK

    @ensure_synched
    def _send_lasso_or_scribble_prompt(self, mask, positive_click=False, tp="lasso"):
        """Uploads lasso or scribble prompt. In ROI mode, crops the mask to the
        window. Skips gzip (server auto-detects), zero-copies bool to uint8,
        skips the redundant empty check (already done by the public entry)."""
        # bool -> uint8 is zero-copy via view; uint8 stays put; anything else falls
        # back to astype.
        if mask.dtype == np.bool_:
            mask_u8 = mask.view(np.uint8)
        elif mask.dtype == np.uint8:
            mask_u8 = mask
        else:
            mask_u8 = mask.astype(np.uint8)

        if self._roi_active:
            send_mask = self._crop_for_roi(mask_u8)
        else:
            send_mask = mask_u8

        url = f"{self.server}/add_{tp}_interaction"
        try:
            buffer = io.BytesIO()
            np.save(buffer, send_mask)
            raw_data = buffer.getvalue()

            from requests_toolbelt import MultipartEncoder

            fields = {
                "file": ("volume.npy", raw_data, "application/octet-stream"),
                "positive_click": str(positive_click),
            }
            encoder = MultipartEncoder(fields=fields)
            seg_response = self.request_to_server(
                url,
                data=encoder,
                headers={"Content-Type": encoder.content_type},
            )

            if seg_response.status_code == 200:
                expected_shape = self._expected_server_shape()
                unpacked_segmentation = self.unpack_binary_segmentation(
                    seg_response.content, decompress=False, vol_shape=expected_shape
                )
                self.show_segmentation(unpacked_segmentation)
            else:
                logging.debug(
                    f"lasso_or_scribble_prompt upload failed with status code: {seg_response.status_code}"
                )
        except Exception as e:
            logging.debug(f"Error in lasso_or_scribble_prompt: {e}")

    def on_scribble_finished(self, caller, event):
        """
        Called when the user completes a scribble stroke in the Paint effect.
        We calculate the diff in the drawn region and send it to the server.
        """
        logging.debug("Scribble stroke finished - labelmap modified!")

        # Clean up observer if you only want it once
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            caller.RemoveObserver(self._scribble_labelmap_callback_tag["tag"])
            label_name = self._scribble_labelmap_callback_tag["label_name"]
            del self._scribble_labelmap_callback_tag
        else:
            return

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.scribble_segment_node, label_name, self.get_volume_node()
        )

        prev_scribble_mask = getattr(self, "_prev_scribble_mask", None)
        if prev_scribble_mask is None:
            # First scribble of the session — no diff, no need to allocate zeros.
            diff_mask = mask
        else:
            diff_mask = mask - prev_scribble_mask
        self._prev_scribble_mask = mask

        with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):
            self.lasso_or_scribble_prompt(
                mask=diff_mask, positive_click=self.is_positive, tp="scribble"
            )

        self.ui.pbInteractionScribble.click()  # turn it off
        self.ui.pbInteractionScribble.click()  # turn it on

    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def make_new_segment(self):
        """
        Creates a new empty segment in the current segmentation, increments a name,
        and sets it as the selected segment.
        """
        # After creating a new segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        # Fresh segment = fresh ROI session: next prompt picks a new window.
        self._reset_roi()

        logging.debug("doing make_new_segment")
        segmentation_node = self.get_segmentation_node()

        # Generate a new segment name
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0:
            new_segment_name = "Segment_1"
        else:
            # Find the next available number
            segment_numbers = [
                int(seg.split("_")[-1])
                for seg in segment_ids
                if seg.startswith("Segment_") and seg.split("_")[-1].isdigit()
            ]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"

        # Create and add the new segment
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            new_segment_name
        )
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Make sure the right node is selected
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment
        and updates the server.
        """
        # After clearing a segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        # Cleared segment = fresh ROI session.
        self._reset_roi()

        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()

        if selected_segment_id:
            logging.debug(f"Clearing segment: {selected_segment_id}")
            self.show_segmentation(
                np.zeros(self.get_image_data().shape, dtype=np.uint8)
            )
            self.setup_prompts()
            self.upload_segment_to_server()
        else:
            logging.debug("No segment selected to clear.")

    def show_segmentation(self, segmentation_mask):
        """
        Updates the currently selected segment with the given binary mask array.
        In ROI mode, segmentation_mask is the native-resolution crop returned by
        the server; we paste it back into the full-volume segment at the crop
        offset (REPLACE within the crop window, untouched outside).
        """
        t0 = time.time()
        if self._roi_active and self._roi_shape_ijk is not None:
            expected_shape = self._expected_server_shape()
            if expected_shape is not None and segmentation_mask.shape == expected_shape:
                segmentation_mask = self._composite_roi_mask_into_full(segmentation_mask)
            else:
                logging.warning(
                    f"ROI mask shape {segmentation_mask.shape} != expected {expected_shape}; "
                    "writing as-is."
                )

        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())

        with slicer.util.RenderBlocker():  # avoid flashing of 3D view
            self.ui.editor_widget.saveStateForUndo()
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segmentation_mask,
                segmentationNode,
                selectedSegmentID,
                self.get_volume_node(),
            )
            if was_3d_shown:
                segmentationNode.CreateClosedSurfaceRepresentation()

        # Mark the segment as being edited (can be useful for selective saving of only modified segments)
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        if segmentation_mask.sum() > 0:
            # If we do this when segmentation_mask.sum() == 0, sometimes Slicer will throw "bogus" OOM errors
            # (see https://github.com/coendevente/SlicerNNInteractive/issues/38)
            segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()

        del segmentation_mask

        # Refresh segment signature so the next selected_segment_changed call
        # doesn't see this server-driven write as a user edit and re-upload.
        self.previous_states["segment_sig"] = (
            id(segmentationNode), selectedSegmentID, int(segmentationNode.GetMTime())
        )

        logging.debug(f"show_segmentation took {time.time() - t0}")

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one.
        """
        # If the segmentation widget has a currently selected segmentation node, return it.
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node:
            if segmentation_node.GetName() != self.scribble_segment_node_name:
                return segmentation_node

        # Otherwise, fall back to getting the first suitable segmentation node
        segmentation_node = None
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segmentation_node in segmentation_nodes:
            if segmentation_node.GetName() == self.scribble_segment_node_name:
                segmentation_node = None
                continue

        # Create new segmentation node if none suitable found
        if not segmentation_node:
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Set segmentation node in widget
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())

        return segmentation_node

    def get_selected_segmentation_node_and_segment_id(self):
        """
        Retrieve the currently selected segmentation node & segment ID.
        If none, create one.
        """
        logging.debug("doing get_selected_segmentation_node_and_segment_id")
        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            return self.make_new_segment()

        return segmentation_node, selected_segment_id

    def get_current_segment_id(self):
        """
        Returns the ID of the segment currently selected in the segment editor.
        """
        return self.ui.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

    def get_segment_data(self):
        """
        Gets the labelmap array (binary) of the currently selected segment.
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, selected_segment_id, self.get_volume_node()
        )
        seg_data_bool = mask.astype(bool)

        return seg_data_bool

    def selected_segment_changed(self):
        """
        Returns True when the server needs the segment re-uploaded. O(1)
        signature: (segmentation node identity, selected segment id, VTK MTime).
        Updated by show_segmentation after each server-driven write so we don't
        falsely flag the just-written-back mask as a user edit on the next prompt.
        """
        seg_node = self.get_segmentation_node()
        seg_id = self.get_current_segment_id()
        if seg_node is None or not seg_id:
            return False
        sig = (id(seg_node), seg_id, int(seg_node.GetMTime()))
        last_sig = self.previous_states.get("segment_sig", None)
        changed = last_sig != sig
        self.previous_states["segment_sig"] = sig
        logging.debug(f"selected_segment_changed: {changed}  sig={sig}")
        return changed

    ###############################################################################
    # Dynamic-ROI helpers (large-volume mode)
    ###############################################################################

    def _volume_spacing_mm(self):
        """Returns (s_i, s_j, s_k) voxel spacing in mm, or None if no volume."""
        vn = self.get_volume_node()
        if vn is None:
            return None
        return tuple(float(s) for s in vn.GetSpacing())

    def _volume_shape_ijk(self):
        """Returns (I, J, K) shape of the current volume, or None."""
        arr = self.get_image_data()
        if arr is None:
            return None
        # numpy array is (K, J, I)
        return tuple(arr.shape[::-1])

    def _should_use_roi_mode(self):
        """Enable when total voxel count exceeds ROI_AUTO_ENABLE_VOXEL_THRESHOLD.
        Above ~1 gigavoxel the server's float32 representation + autozoom
        replicas blow past a 16 GB GPU."""
        if FORCE_ROI_MODE:
            return True
        vol_ijk = self._volume_shape_ijk()
        if vol_ijk is None:
            return False
        total = int(vol_ijk[0]) * int(vol_ijk[1]) * int(vol_ijk[2])
        return total > ROI_AUTO_ENABLE_VOXEL_THRESHOLD

    def _compute_window_shape(self):
        """Aspect-preserving crop window sized to ROI_BUDGET_VOXEL_CUBE^3 voxels.
        Each axis gets vol_axis × k where k = (BUDGET / prod(vol))^(1/3),
        clamped to vol_axis. Returns shape_ijk."""
        vol_ijk = self._volume_shape_ijk()
        if vol_ijk is None:
            return None
        prod = int(vol_ijk[0]) * int(vol_ijk[1]) * int(vol_ijk[2])
        budget = ROI_BUDGET_VOXEL_CUBE ** 3
        k = (budget / prod) ** (1.0 / 3.0)
        shape = tuple(
            min(vol_ijk[a], max(1, int(round(vol_ijk[a] * k))))
            for a in range(3)
        )
        return shape

    def _compute_roi_for_center(self, center_ijk):
        """Center the crop window on the given IJK point, clipped to volume
        bounds. Returns (start_ijk, shape_ijk)."""
        vol_ijk = self._volume_shape_ijk()
        if vol_ijk is None:
            return None
        shape_ijk = self._compute_window_shape()
        if shape_ijk is None:
            return None
        start_ijk = []
        for axis in range(3):
            extent = shape_ijk[axis]
            half = extent // 2
            start = int(center_ijk[axis]) - half
            if start < 0:
                start = 0
            if start + extent > vol_ijk[axis]:
                start = vol_ijk[axis] - extent
            if start < 0:
                start = 0
            start_ijk.append(int(start))
        return tuple(start_ijk), tuple(shape_ijk)

    def _in_current_roi(self, ijk):
        """True if ijk is anywhere inside the current crop window."""
        if not self._roi_active or self._roi_start_ijk is None:
            return False
        for axis in range(3):
            if ijk[axis] < self._roi_start_ijk[axis]:
                return False
            if ijk[axis] >= self._roi_start_ijk[axis] + self._roi_shape_ijk[axis]:
                return False
        return True

    def _in_comfort_zone(self, ijk):
        """True if ijk falls in the central ROI_COMFORT_FRACTION of the crop
        window — the margin from each face is window_size × (1 - frac) / 2.
        Outside this zone we recenter so the model always has headroom."""
        if not self._roi_active or self._roi_start_ijk is None:
            return False
        for axis in range(3):
            size = self._roi_shape_ijk[axis]
            margin = int(size * (1.0 - ROI_COMFORT_FRACTION) / 2.0)
            lo = self._roi_start_ijk[axis] + margin
            hi = self._roi_start_ijk[axis] + size - margin
            if ijk[axis] < lo or ijk[axis] >= hi:
                return False
        return True

    def _ijk_to_roi_local(self, ijk):
        """Translate an IJK in the original volume to ROI-local IJK (native)."""
        if not self._roi_active:
            return tuple(int(v) for v in ijk)
        return tuple(int(ijk[a] - self._roi_start_ijk[a]) for a in range(3))

    def _maybe_update_roi_for_prompt(self, anchor_ijk):
        """Decide whether to (re)establish ROI for a prompt at anchor_ijk.
        Returns True if ROI state changed (caller's image_changed will pick it
        up and trigger re-upload)."""
        if anchor_ijk is None:
            return False
        if not self._should_use_roi_mode():
            if self._roi_active:
                self._reset_roi()
                return True
            return False

        if self._roi_active and self._in_comfort_zone(anchor_ijk):
            return False  # Prompt is inside the comfort zone; reuse current crop

        new = self._compute_roi_for_center(anchor_ijk)
        if new is None:
            return False
        start, shape = new
        if (
            self._roi_active
            and self._roi_start_ijk == start
            and self._roi_shape_ijk == shape
        ):
            return False
        self._roi_active = True
        self._roi_start_ijk = start
        self._roi_shape_ijk = shape
        logging.info(
            f"ROI crop window: start_ijk={start}, shape_ijk={shape}, "
            f"upload_shape_kji={tuple(shape[::-1])}"
        )
        self._refresh_roi_status_label()
        return True

    def _reset_roi(self):
        """Reset ROI state so the next prompt establishes a fresh ROI."""
        self._roi_active = False
        self._roi_start_ijk = None
        self._roi_shape_ijk = None
        self._refresh_roi_status_label()

    def _crop_for_roi(self, arr):
        """arr is (K, J, I) numpy. Returns the native-resolution crop."""
        i0, j0, k0 = self._roi_start_ijk
        di, dj, dk = self._roi_shape_ijk
        sub = arr[k0:k0 + dk, j0:j0 + dj, i0:i0 + di]
        return np.ascontiguousarray(sub)

    def _expected_server_shape(self):
        """Numpy (K, J, I) shape the server will use for the active ROI."""
        if not self._roi_active:
            return None
        di, dj, dk = self._roi_shape_ijk
        return (dk, dj, di)

    def _compute_max_zoom_out_factor(self):
        """Compute (or fetch override) the autozoom cap to send to the server.
        Based on spacing only — there is no client-side downsample."""
        if self._roi_max_zoom_out_factor_override is not None:
            return float(self._roi_max_zoom_out_factor_override)
        spacing = self._volume_spacing_mm()
        if spacing is None:
            return None
        min_spacing = min(spacing)
        if min_spacing <= 0:
            return None
        factor = ROI_REFERENCE_SPACING_MM / min_spacing
        factor = max(1.0, min(ROI_MAX_ZOOM_OUT_FACTOR_CAP, factor))
        return float(np.ceil(factor))

    def _composite_roi_mask_into_full(self, roi_mask):
        """roi_mask is native-resolution, shape matches _roi_shape_ijk (KJI).
        REPLACE the crop region of the full-volume segment with roi_mask;
        voxels outside the crop region are left untouched."""
        arr = self.get_image_data()
        if arr is None:
            return roi_mask
        i0, j0, k0 = self._roi_start_ijk
        di, dj, dk = self._roi_shape_ijk
        # Make sure incoming mask is exactly the crop shape
        roi_mask = roi_mask[:dk, :dj, :di]
        existing = self.get_segment_data().astype(np.uint8)
        if existing.shape != arr.shape:
            existing = np.zeros(arr.shape, dtype=np.uint8)
        existing[k0:k0 + dk, j0:j0 + dj, i0:i0 + di] = roi_mask.astype(np.uint8)
        return existing

    ###############################################################################
    # Server communication and sync functions
    ###############################################################################

    def update_server(self):
        """
        Reads user-entered server URL from UI, saves to QSettings, updates self.server.
        Only applies in external server mode.
        """
        if self._internal_server_mode:
            return
        self.server = self.ui.Server.text.rstrip("/")
        settings = qt.QSettings()
        settings.setValue("SlicerNNInteractive/server", self.server)
        logging.debug(f"Server URL updated and saved: {self.server}")

    def test_server_connection(self):
        """
        Sends a lightweight GET request to see if the configured server responds.
        """
        server_text = self.ui.Server.text
        if not server_text.strip():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Test Connection",
                "Please enter a server URL before testing the connection.",
            )
            return

        self.ui.Server.setText(server_text.strip())
        self.update_server()
        server_url = self.server

        if getattr(self, "_test_server_in_progress", False):
            return
        self._test_server_in_progress = True

        slicer.util.showStatusMessage("Testing nnInteractive server connection...", 2000)
        slicer.app.processEvents()

        response = None
        error_message = None
        try:
            response = requests.get(server_url, timeout=5)
        except requests.exceptions.MissingSchema:
            error_message = (
                "Server URL is invalid. Make sure it starts with 'http://' or 'https://'."
            )
        except requests.exceptions.RequestException as exc:
            error_message = str(exc)
        finally:
            self._test_server_in_progress = False
            slicer.util.showStatusMessage("")

        if response is not None:
            info_message = (
                f"Server at '{server_url}' is reachable."
            )
            QMessageBox.information(
                slicer.util.mainWindow(),
                "Test Connection",
                info_message,
            )
            return
        else:
            QMessageBox.critical(
                slicer.util.mainWindow(),
                "Test Connection",
                f"Failed to reach '{server_url}'.\n\n{error_message}",
            )

    def request_to_server(self, *args, **kwargs):
        """
        Wraps requests.post in a try/except and shows error in pop up windows if necessary.
        """

        with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):

            error_message = None
            try:
                response = requests.post(*args, **kwargs)
                logging.debug(f"response: {response}")
            except requests.exceptions.MissingSchema as e:
                response = None
                if self.server == "":
                    raise RuntimeError("It seems you have not set the server URL yet. You can configure it in the 'Configuration' tab.")
                else:
                    raise RuntimeError(f"Server URL '{self.server}' is unreachable. You can edit the URL in the 'Configuration' tab.")
            except requests.exceptions.ConnectionError as e:
                response = None
                raise RuntimeError(f"Failed to connect to server '{self.server}'. Please make sure the server is running and check the server URL in the 'Configuration' tab.")
            except requests.exceptions.InvalidSchema as e:
                append_text_to_error_message = ""
                if not args[0].startswith("http://"):
                    append_text_to_error_message = "\n\nHint: Perhaps your Server URL in the 'Configuration' tab should start with 'http://'. For example, if your server runs on localhost and port 1527, 'localhost:1527' would not work as a Server URL, while 'http://localhost:1527' would."
                raise RuntimeError(f'{e}{append_text_to_error_message}')

            if response.status_code != 200:
                status_code = response.status_code
                response = None
                raise RuntimeError(f"Something has gone wrong with your request (Status code {status_code}).")

            t0 = time.time()
            # Try to parse JSON and check for a specific error.
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                resp_json = response.json()
                if resp_json.get("status") == "error":
                    if "No image uploaded" in resp_json.get("message", ""):
                        logging.debug("No image has been uploaded to the server. Please upload an image first.")
                        self.upload_image_to_server()
                        self.upload_segment_to_server()
                        return self.request_to_server(*args, **kwargs)
                    else:
                        response = None
                        raise RuntimeError(f"Server error: {resp_json.get('message', 'Unknown error')}")

            logging.debug(f"1157 took {time.time() - t0}")

        return response

    def upload_image_to_server(self):
        """
        Gets volume data from Slicer, packs it, and uploads it to the server.
        In ROI mode, sends only the native-resolution crop window around the
        active prompt, plus max_zoom_out_factor for the server's autozoom.
        """
        logging.debug("Syncing image with server...")
        try:
            t0 = time.time()
            image_data = self.get_image_data()
            logging.debug(f"self.get_image_data took {time.time() - t0}")

            if image_data is None:
                logging.debug("No image data available to upload.")
                return

            if self._roi_active:
                upload_arr = self._crop_for_roi(image_data)
                logging.info(
                    f"ROI image upload: full shape {image_data.shape} -> "
                    f"crop shape {upload_arr.shape} (native res)"
                )
            else:
                upload_arr = image_data

            url = f"{self.server}/upload_image"

            buffer = io.BytesIO()
            np.save(buffer, upload_arr)
            raw_data = buffer.getvalue()
            logging.debug(f"len(raw_data): {len(raw_data)}")

            fields = {"file": ("volume.npy", raw_data, "application/octet-stream")}
            mzof = self._compute_max_zoom_out_factor()
            if mzof is not None:
                fields["max_zoom_out_factor"] = str(mzof)
                logging.info(f"Sending max_zoom_out_factor={mzof}")

            from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

            slicer.progress_window = slicer.util.createProgressDialog(autoClose=False)
            slicer.progress_window.minimum = 0
            slicer.progress_window.maximum = 100
            slicer.progress_window.setLabelText("Uploading image...")

            def my_callback(monitor):
                if not hasattr(monitor, "last_update"):
                    monitor.last_update = time.time()
                if time.time() - monitor.last_update <= 0.2:
                    return
                monitor.last_update = time.time()
                slicer.progress_window.setValue(
                    monitor.bytes_read / len(raw_data) * 100
                )
                slicer.progress_window.show()
                slicer.progress_window.activateWindow()
                slicer.progress_window.setLabelText("Uploading image...")
                slicer.app.processEvents()

            encoder = MultipartEncoder(fields=fields)
            monitor = MultipartEncoderMonitor(encoder, my_callback)

            try:
                result = self.request_to_server(
                    url, data=monitor, headers={"Content-Type": monitor.content_type}
                )
            finally:
                slicer.progress_window.close()

            return result
        except Exception as e:
            logging.debug(f"Error in upload_image_to_server: {e}")

    def upload_segment_to_server(self):
        """
        Grabs current segmentation labelmap and uploads it. In ROI mode, crops
        the segment to the same native-resolution window as the uploaded image.

        Performance notes vs prior PR code:
          - `.view(np.uint8)` instead of `.astype(np.uint8)` is zero-copy
            (numpy bool is 1 byte) — saves a full-volume allocation that on a
            multi-GB segment was costing seconds.
          - Skip gzip entirely. On localhost there is no bandwidth reason to
            compress, and gzip on a 134 MB cropped mask costs ~1-2 s per upload.
        """
        logging.debug("Syncing segment with server...")
        try:
            segment_data = self.get_segment_data()  # bool, full-volume
            if self._roi_active:
                segment_data = self._crop_for_roi(segment_data.view(np.uint8))
                logging.info(f"ROI segment upload shape: {segment_data.shape}")
            else:
                segment_data = segment_data.view(np.uint8)

            buffer = io.BytesIO()
            np.save(buffer, segment_data)
            files = {"file": ("volume.npy", buffer.getvalue(), "application/octet-stream")}

            url = f"{self.server}/upload_segment"
            result = self.request_to_server(url, files=files)
            return result
        except Exception as e:
            logging.debug(f"Error in upload_segment_to_server: {e}")

    ###############################################################################
    # Utility / converters functions
    ###############################################################################

    def get_image_data(self):
        """
        Returns the voxel data of the current active (or first available) volume node.
        """
        volume_node = self.get_volume_node()
        if volume_node:
            return slicer.util.arrayFromVolume(volume_node)

        return None

    def get_volume_node(self):
        """
        Retrieves the current source volume node chosen in the segment editor widget.
        If nothing is set then use the most recently added scalar volume
        """
        # Get volume node from segment editor widget
        volumeNode = self.ui.editor_widget.sourceVolumeNode()

        if not volumeNode:
            # Get the most recently added volume node
            volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if volumeNodes:
                volumeNode = volumeNodes[-1]
            # Show this volume node in the segment editor widget
            self.ui.editor_widget.setSourceVolumeNode(volumeNode)

        return volumeNode

    def image_changed(self, do_prev_image_update=True):
        """
        Returns True when the server needs a (re-)uploaded image. Uses an O(1)
        signature: (volume node identity, VTK MTime, ROI window, zoom factor).
        No byte-level array comparison — the previous full-volume `np.array_equal`
        + `copy.deepcopy` cost seconds per prompt on multi-GB volumes for zero
        benefit; Slicer/VTK bumps MTime on every modification.
        """
        vol_node = self.get_volume_node()
        if vol_node is None:
            logging.debug("No volume node found")
            return

        sig = (
            id(vol_node),
            int(vol_node.GetMTime()),
            self._roi_active,
            self._roi_start_ijk,
            self._roi_shape_ijk,
            self._compute_max_zoom_out_factor(),
        )
        last_sig = self.previous_states.get("upload_sig", None)
        changed = last_sig != sig

        if do_prev_image_update:
            self.previous_states["upload_sig"] = sig

        return changed

    def mask_to_np_upload_file(self, mask):
        """
        Converts a numpy mask into a gzipped file object for POSTing.
        """
        buffer = io.BytesIO()
        np.save(buffer, mask)
        compressed_data = gzip.compress(buffer.getvalue())

        files = {"file": ("volume.npy.gz", compressed_data, "application/octet-stream")}

        return files

    def unpack_binary_segmentation(self, binary_data, decompress=False, vol_shape=None):
        """
        Unpacks data received from server into a 3D numpy array (uint8 0/1).
        If vol_shape is None, infers from the current volume node (legacy path).
        In ROI mode, callers pass the expected crop-window shape.
        """
        if decompress:
            binary_data = gzip.decompress(binary_data)

        if vol_shape is None:
            if self.get_image_data() is None:
                self.capture_image()
            vol_shape = self.get_image_data().shape

        total_voxels = int(np.prod(vol_shape))
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        unpacked_bits = unpacked_bits[:total_voxels]

        segmentation_mask = (
            unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
        )

        return segmentation_mask

    def ras_to_xyz(self, pos):
        """
        Converts an RAS position to IJK voxel coords in the current volume node.
        """
        volumeNode = self.get_volume_node()

        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas
        )
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)

        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
        xyz = [int(round(c)) for c in point_Ijk[0:3]]
        return xyz


    def xyz_from_caller(self, caller, lock_point=True, point_type="control_point"):
        """
        Extract voxel coordinates from a Markups node.
        `point_type` can be either "control_point" or "curve_point".
        """
        if point_type == "control_point":
            n = caller.GetNumberOfControlPoints()
            if n < 0:
                logging.debug("No control points found")
                return

            pos = [0, 0, 0]
            caller.GetNthControlPointPosition(n - 1, pos)
            if lock_point:
                caller.SetNthControlPointLocked(n - 1, True)
            xyz = self.ras_to_xyz(pos)
            return xyz
        elif point_type == "curve_point":
            vtk_pts = caller.GetCurvePointsWorld()
            
            if vtk_pts is not None:
                vtk_pts_data = vtk_to_numpy(vtk_pts.GetData())
                xyz = [self.ras_to_xyz(pos) for pos in vtk_pts_data]
                logging.debug(xyz)
                return xyz

            return []
        else:
            raise ValueError(f'Unknown point_type {point_type}')

    def lasso_points_to_mask(self, points):
        """
        Given a list of voxel coords (defining a polygon in one slice),
        returns a 3D mask with that polygon filled in the appropriate slice.
        """
        from skimage.draw import polygon

        shape = self.get_image_data().shape
        pts = np.array(points)  # shape (n, 3)

        # Determine which coordinate is constant
        const_axes = [i for i in range(3) if np.unique(pts[:, i]).size == 1]
        if len(const_axes) != 1:
            raise ValueError(
                "Expected exactly one constant coordinate among the points"
            )
        const_axis = const_axes[0]
        const_val = int(pts[0, const_axis])

        # Create a blank 3D mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Depending on which axis is constant, extract the 2D polygon and fill the corresponding slice.
        # Note: our volume is ordered as (z, y, x)
        if const_axis == 2:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            rr, cc = polygon(y_coords, x_coords, shape=(shape[1], shape[2]))
            mask[const_val, rr, cc] = 1
        elif const_axis == 1:
            x_coords = pts[:, 0]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, x_coords, shape=(shape[0], shape[2]))
            mask[rr, const_val, cc] = 1
        elif const_axis == 0:
            y_coords = pts[:, 1]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, y_coords, shape=(shape[0], shape[1]))
            mask[rr, cc, const_val] = 1

        return mask

    ###############################################################################
    # Prompt type toggle (positive / negative)
    ###############################################################################

    @property
    def is_positive(self):
        """
        Returns True if the current prompt is set to "positive",
        False if "negative."
        """
        return self.ui.pbPromptTypePositive.isChecked()

    def on_prompt_type_positive_clicked(self, checked=False):
        """
        Called when user presses the "Positive" prompt button.
        """
        # Update UI
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypePositive.setChecked(True)
        self.ui.pbPromptTypeNegative.setChecked(False)
        logging.debug("Prompt type set to POSITIVE")

    def on_prompt_type_negative_clicked(self, checked=False):
        """
        Called when user presses the "Negative" prompt button.
        """

        # Update UI
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypePositive.setChecked(False)
        self.ui.pbPromptTypeNegative.setChecked(True)
        logging.debug("Prompt type set to NEGATIVE")

    def toggle_prompt_type(self, checked=False):
        """
        Toggle between positive and negative (triggered by 'T' key).
        """
        logging.debug("Toggling prompt type (positive <> negative)")
        if self.current_prompt_type_positive:
            self.on_prompt_type_negative_clicked()
        else:
            self.on_prompt_type_positive_clicked()

    ensure_synched = staticmethod(ensure_synched)


###############################################################################
# Test hook (used by Reload & Test)
###############################################################################
_test_module_path = (
    Path(__file__).resolve().parents[0]
    / "Testing"
    / "Python"
    / "SlicerNNInteractiveSegmentationTest.py"
)

if _test_module_path.exists():
    import importlib.util as _importlib_util

    _spec = _importlib_util.spec_from_file_location(
        "SlicerNNInteractiveSegmentationTest", str(_test_module_path)
    )
    _test_module = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_test_module)
    SlicerNNInteractiveTest = _test_module.SlicerNNInteractiveSegmentationTest
