import time
import warnings
import os
import io
import gzip
import hashlib
import argparse

import numpy as np
import torch
import uvicorn

import xxhash

from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import (
    snapshot_download,
)

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from torch.nn.functional import interpolate
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.utils.crop import paste_tensor
from nnunetv2.utilities.helpers import dummy_context, empty_cache


from fastapi import FastAPI, Response, UploadFile, File, Form


###############################################################################
# Global constants & FastAPI app
###############################################################################
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".nninteractive_weights")
DEFAULT_MAX_ZOOM_OUT = 4.0  # nnInteractive's hardcoded cap; we make it overridable

app = FastAPI()


###############################################################################
# Subclass with configurable max_zoom_out_factor
###############################################################################
# Upstream nnInteractive hardcodes the autozoom cap to 4 in three spots inside
# _predict. We copy the method to make the cap an instance attribute. This is
# fragile w.r.t. upstream changes; if nnInteractive's _predict body changes
# meaningfully, re-sync this override.
class ConfigurableZoomSession(nnInteractiveInferenceSession):
    max_zoom_out_factor: float = DEFAULT_MAX_ZOOM_OUT

    def _predict(self, force_full_refine: bool = False):
        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        if len(self.new_interaction_centers) == 0:
            print('No patch queued for prediction. Nothing to do.')
            return

        if len(self.new_interaction_centers) > 1:
            print('It seems like more than one interaction was added since the last prediction. This is not '
                  'recommended and may cause unexpected behavior or inefficient predictions\n'
                  '!!!WE NO LONGER RUN ONE PREDICTION PER CENTER AND ONLY USE THE LAST ADDED INTERACTION AS CENTER!!!')

        cap = float(self.max_zoom_out_factor)
        prediction_center = self.new_interaction_centers[-1]
        zoom_out_factor = min(cap, self.new_interaction_zoom_out_factors[-1])

        start_predict = time.time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            start_initial_pred = time.time()
            input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(prediction_center, zoom_out_factor)
            pred = self.network(input_for_predict[None])[0].argmax(0).detach()
            del input_for_predict

            previous_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox)
            if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
                previous_prediction = interpolate(
                    previous_prediction[None, None].to(float), pred.shape, mode='nearest'
                )[0, 0]
            has_change = self._detect_change_at_border(pred, previous_prediction)
            del previous_prediction

            print(f'Took {round(time.time() - start_initial_pred, 3)} s for initial prediction at zoom out factor {zoom_out_factor}')

            zoom_out_growth_factor = 1.5
            start_zoomout = time.time()
            while has_change and self.do_autozoom:
                print(f'AutoZoom zoom out factor {zoom_out_factor} (cap {cap})')
                if zoom_out_factor >= cap:
                    break
                zoom_out_factor *= zoom_out_growth_factor
                zoom_out_factor = min(cap, zoom_out_factor)

                input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(prediction_center, zoom_out_factor)
                pred = self.network(input_for_predict[None])[0].argmax(0).detach()
                del input_for_predict

                previous_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox)
                if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
                    previous_prediction_resized = interpolate(
                        previous_prediction[None, None].to(float), pred.shape, mode='nearest'
                    )[0, 0]
                else:
                    previous_prediction_resized = previous_prediction
                has_change = self._detect_change_at_border(pred, previous_prediction_resized)

            if zoom_out_factor > 1:
                print(f'Zoom out took {round(time.time() - start_zoomout, 3)} s, max zoom out factor {zoom_out_factor}')
            else:
                print('No zoom out necessary')

            if zoom_out_factor == 1:
                paste_tensor(self.interactions[0], pred.half(), scaled_bbox)
                bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                        zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                paste_tensor(
                    self.target_buffer,
                    pred.to(self.target_buffer.device) if isinstance(self.target_buffer, torch.Tensor) else pred.to('cpu'),
                    bbox,
                )
                print('No refinement necessary')
            else:
                prediction_with_coarse = self.interactions[0]
                if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                    pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                diff_map, has_diff = self._compute_diff_map(pred, self.interactions[0], scaled_bbox, scaled_patch_size)
                if force_full_refine:
                    print('Forcing full refinement of entire structure')
                    diff_map[self.interactions[0] > 0] = 1
                paste_tensor(prediction_with_coarse, pred, scaled_bbox)
                self._refine_coarse(diff_map, prediction_with_coarse)
                del prediction_with_coarse

        print(f'Done. Total time {round(time.time() - start_predict, 3)}s')
        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

###############################################################################
# Utility / helper functions
###############################################################################


def calculate_md5_array(image_data, xx=False):
    """
    Calculate either an xxHash (if xx=True) or MD5 hash of a NumPy array's bytes.
    """
    if xx:
        xh = xxhash.xxh64()
        xh.update(image_data.tobytes())

        out_hash = xh.hexdigest()
    else:
        md5_hash = hashlib.md5()
        md5_hash.update(image_data.tobytes())
        out_hash = md5_hash.hexdigest()

    return out_hash


def unpack_binary_segmentation(binary_data, vol_shape):
    """
    Unpacks binary data (1 bit per voxel) into a full 3D numpy array (bool type).
    """
    total_voxels = np.prod(vol_shape)
    unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
    unpacked_bits = unpacked_bits[:total_voxels]
    segmentation_mask = (
        unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
    )

    return segmentation_mask


def segmentation_binary(seg_in, compress=False):
    """
    Convert a (boolean) segmentation array into packed bits and optionally compress.
    """
    seg_result = seg_in.astype(bool)  # Convert to bool type if not already
    packed_segmentation = np.packbits(seg_result, axis=None)  # Pack into 1D byte array
    packed_segmentation = packed_segmentation.tobytes()
    if compress:
        packed_segmentation = gzip.compress(packed_segmentation)
    return packed_segmentation  # Convert to bytes for transmission


def process_mask_and_click_input(file_bytes, positive_click):
    """
    Helper that loads the numpy mask (auto-decompressing if gzip-framed) and
    interprets the positive_click string as a boolean. Accepts both gzip
    (magic 0x1f 0x8b) and raw npy so updated and legacy clients both work.
    """
    positive_click_bool = positive_click.lower() in ["true", "1", "yes"]

    error = get_error_if_img_not_set()
    if error is not None:
        return error

    if len(file_bytes) >= 2 and file_bytes[:2] == b"\x1f\x8b":
        try:
            file_bytes = gzip.decompress(file_bytes)
        except Exception as e:
            return {"status": "error", "message": f"Decompression failed: {e}"}

    mask = np.load(io.BytesIO(file_bytes))

    return mask, positive_click_bool


def get_error_if_img_not_set():
    if PROMPT_MANAGER is None or PROMPT_MANAGER.img is None:
        warnings.warn("There is no image in the server. Be sure to send it before")
        return {"status": "error", "message": "No image uploaded"}

    return


###############################################################################
# PromptManager class
###############################################################################
class PromptManager:
    """
    Manages the image, target tensor, and runs inference sessions for point, bbox,
    lasso, and scribble interactions.
    """

    def __init__(self):
        self.img = None
        self.target_tensor = None

        self.download_weights()
        self.session = self.make_session()

    def download_weights(self):
        """
        Downloads only the files matching 'MODEL_NAME/*' into DOWNLOAD_DIR.
        """
        snapshot_download(
            repo_id=REPO_ID, allow_patterns=[f"{MODEL_NAME}/*"], local_dir=DOWNLOAD_DIR
        )

    def make_session(self):
        """
        Creates an nnInteractiveInferenceSession, points it at the downloaded model.
        """
        session = ConfigurableZoomSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=True,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )

        # Load the trained model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)

        return session

    def set_image(self, input_image, max_zoom_out_factor=None):
        """
        Loads the user-provided 3D image into the session, resets interactions.
        Optionally overrides the autozoom cap for this image.
        """
        self.session.reset_interactions()

        if max_zoom_out_factor is not None and max_zoom_out_factor > 0:
            self.session.max_zoom_out_factor = float(max_zoom_out_factor)
        else:
            self.session.max_zoom_out_factor = DEFAULT_MAX_ZOOM_OUT
        print(f"Session max_zoom_out_factor set to {self.session.max_zoom_out_factor}")

        self.img = input_image[None]  # Ensure shape (1, x, y, z)
        self.session.set_image(self.img)

        print("self.img.shape:", self.img.shape)

        # Validate input dimensions
        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(
            self.img.shape[1:], dtype=torch.uint8
        )  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)

    def set_segment(self, mask):
        """
        Sets or resets a segmentation (mask) on the server side.
        If mask is empty, resets the session's interactions.
        """
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(
                self.img.shape[1:], dtype=torch.uint8
            )  # Must be 3D (x, y, z)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(mask)

    def add_point_interaction(self, point_coordinates, include_interaction):
        """
        Process a point-based interaction (positive or negative).
        """
        self.session.add_point_interaction(
            point_coordinates, include_interaction=include_interaction
        )

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_bbox_interaction(
        self, outer_point_one, outer_point_two, include_interaction
    ):
        """
        Process bounding box-based interaction.
        """
        print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)

        data = np.array([outer_point_one, outer_point_two])
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)

        bbox = [
            [int(_min[0]), int(_max[0])],
            [int(_min[1]), int(_max[1])],
            [int(_min[2]), int(_max[2])],
        ]

        # Call the session's bounding box interaction function.
        self.session.add_bbox_interaction(bbox, include_interaction=include_interaction)

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_lasso_interaction(self, mask, include_interaction):
        """
        Process lasso-based interaction using a 3D mask.
        """
        print("Lasso mask received with shape:", mask.shape)
        self.session.add_lasso_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_scribble_interaction(self, mask, include_interaction):
        """
        Process scribble-based interaction using a 3D mask.
        """
        print("Scribble mask received with shape:", mask.shape)
        self.session.add_scribble_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()


###############################################################################
# Global prompt manager instance (initialized on startup)
###############################################################################
PROMPT_MANAGER = None


@app.on_event("startup")
async def startup_event():
    global PROMPT_MANAGER
    PROMPT_MANAGER = PromptManager()


###############################################################################
# FastAPI endpoints
###############################################################################


#
# -- Upload endpoints
#
@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(None),
    max_zoom_out_factor: float = Form(None),
):
    """
    Receive a npy file from the client and set it as the main image in PromptManager.
    Optionally accept max_zoom_out_factor as a form field to override the autozoom cap.
    """
    file_bytes = await file.read()
    arr = np.load(io.BytesIO(file_bytes))
    PROMPT_MANAGER.set_image(arr, max_zoom_out_factor=max_zoom_out_factor)

    return {"status": "ok"}


@app.post("/upload_segment")
async def upload_segment(
    file: UploadFile = File(None),
):
    """
    Receive an npy file from the client and set it as the segmentation in PromptManager.
    Accepts both gzip-compressed and raw npy (auto-detected via magic bytes) so
    clients that still send gzip continue to work.
    """
    error = get_error_if_img_not_set()
    if error is not None:
        return error

    file_bytes = await file.read()
    # gzip magic bytes are 0x1f 0x8b; npy magic starts with 0x93 'N' 'U' 'M' 'P' 'Y'.
    if len(file_bytes) >= 2 and file_bytes[:2] == b"\x1f\x8b":
        file_bytes = gzip.decompress(file_bytes)
    arr = np.load(io.BytesIO(file_bytes))

    PROMPT_MANAGER.set_segment(arr)
    return {"status": "ok"}


#
# -- Point interaction endpoint
#
class PointParams(BaseModel):
    voxel_coord: list[int]
    positive_click: bool


@app.post("/add_point_interaction")
async def add_point_interaction(params: PointParams):
    """
    Receives a point (voxel_coord) + positive/negative. Updates the model & returns a binary mask.
    """
    error = get_error_if_img_not_set()
    if error is not None:
        return error
    
    t = time.time()

    seg_result = PROMPT_MANAGER.add_point_interaction(
        point_coordinates=params.voxel_coord, include_interaction=params.positive_click
    )
    compressed_bin = segmentation_binary(seg_result, compress=False)
    print(f"Server whole infer function time: {time.time() - t}")

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
    )


#
# -- Bounding Box interaction endpoint
#
class BBoxParams(BaseModel):
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_click: bool


@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    """
    Receives bounding box corners + positive/negative. Updates model & returns a mask.
    """
    error = get_error_if_img_not_set()
    if error is not None:
        return error
    
    t = time.time()

    seg_result = PROMPT_MANAGER.add_bbox_interaction(
        params.outer_point_one,
        params.outer_point_two,
        include_interaction=params.positive_click,
    )

    segmentation_binary_data = segmentation_binary(seg_result, compress=False)
    print(f"Server whole infer function time: {time.time() - t}")

    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
    )


#
# -- Lasso interaction endpoint
#


@app.post("/add_lasso_interaction")
async def add_lasso_interaction(
    file: UploadFile = File(...), positive_click: str = Form(...)
):
    """
    Receives a gzipped npy mask + positive/negative. Treated as a 'lasso' 3D mask.
    """
    error = get_error_if_img_not_set()
    if error is not None:
        return error
    
    file_bytes = await file.read()
    mask, positive_click_bool = process_mask_and_click_input(file_bytes, positive_click)

    # Process the lasso interaction.
    seg_result = PROMPT_MANAGER.add_lasso_interaction(
        mask, include_interaction=positive_click_bool
    )

    # Convert the segmentation result to compressed binary data.
    segmentation_binary_data = segmentation_binary(seg_result, compress=False)

    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
    )


#
# -- Scribble interaction endpoint
#
@app.post("/add_scribble_interaction")
async def add_scribble_interaction(
    file: UploadFile = File(...), positive_click: str = Form(...)
):
    """
    Receives a scribble mask + positive/negative. Updates model, returns updated segmentation.
    """
    error = get_error_if_img_not_set()
    if error is not None:
        return error
    
    # Read the uploaded file bytes and decompress.
    file_bytes = await file.read()

    mask, positive_click_bool = process_mask_and_click_input(file_bytes, positive_click)

    seg_result = PROMPT_MANAGER.add_scribble_interaction(
        mask, include_interaction=positive_click_bool
    )

    # Convert the segmentation result to compressed binary data.
    segmentation_binary_data = segmentation_binary(seg_result, compress=False)

    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
    )


def main():
    global DOWNLOAD_DIR
    parser = argparse.ArgumentParser(description="Run the nnInteractive Slicer server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    parser.add_argument(
        "--weights-dir",
        default=DOWNLOAD_DIR,
        help="Directory for model weights (default: ~/.nninteractive_weights).",
    )
    args = parser.parse_args()
    DOWNLOAD_DIR = args.weights_dir

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    print(f"torch.__version__: {torch.__version__}")
    main()
