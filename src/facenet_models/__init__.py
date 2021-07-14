import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import crop_resize


class FacenetModel:
    def __init__(self, device=None):
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.lower() not in {"cuda", "cpu"}:
                raise ValueError(f"Expected `device` to be one of (None, 'cuda', 'cpu') but got {device}")
            if device.lower() == "cuda" and not torch.cuda.is_available():
                print("Cuda is not available; falling back to CPU")
                device = "cpu"
            self._device = device.lower()
        self._mtcnn = MTCNN()
        self._resnet = InceptionResnetV1(pretrained="vggface2", device=self._device).eval()

    def detect(self, image):
        """ Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray, shape=(R, C, 3)
            The image in which to detect faces.

        Returns
        -------
        Tuple[np.ndarray shape=(N, 4), np.ndarray shape=(N,), np.ndarray shape=(N, 5)]
            (boxes, probabilities, landmarks) where:
            - boxes is a shape-(N, 4) array of boxes, where N is the number of faces detected in the image.
              Each box is represented as (left, top, right, bottom).
            - probabilities is a shape-(N,) array of probabilities corresponding to each detected face.
            - landmarks is a shape-(N, 5) arrays of facial landmarks corresponding to each detected face.
        """
        return self._mtcnn.detect(np.ascontiguousarray(image), landmarks=True)

    def compute_descriptors(self, image, boxes):
        """ Compute descriptor vectors for the faces contained in ``boxes``.

        Parameters
        ----------
        image : np.ndarray, shape=(R, C, 3)
            The image in which to detect faces.

        boxes : np.ndarray, shape=(N, 4)
            The bounding boxes containing the faces for which to compute descriptors. The boxes should be
            given as (left, top, right, bottom).

        Returns
        -------
        np.ndarray, shape=(N, 512)
            The descriptor vectors, where N is the number of faces.
        """
        crops = [crop_resize(image, [int(max(0, coord)) for coord in box], 160) for box in boxes]
        crops = (torch.tensor(crops).float() - 127.5) / 128
        with torch.no_grad():
            return self._resnet(crops.permute(0, 3, 1, 2).to(self._device)).cpu().numpy()
