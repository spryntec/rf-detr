import torch, cv2

class RRBoxAnnotator(BaseAnnotator):
    """
    A class for drawing oriented bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        self.color = color
        self.thickness = thickness
        self.color_lookup = color_lookup

    def box_xyxy_to_cxcywh(self, x: torch.Tensor) -> torch.Tensor:
        x0, y0, x1, y1 = x.unbind(-1)
        return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)

    def rbox_cxcywh_to_xyxy(self, cxcy: torch.Tensor, offset: float = 0.5) -> torch.Tensor:
        x_c, y_c, w, h = cxcy.unbind(-1)
        x_c -= offset
        y_c -= offset
        R = torch.sqrt(x_c ** 2 + y_c ** 2)
        R[R == 0] = 1e-4  # prevent division by zero

        cosine = x_c / R
        sine = y_c / R

        left_x = x_c + (w / 2) * sine
        left_y = y_c - (w / 2) * cosine
        right_x = x_c - (w / 2) * sine
        right_y = y_c + (w / 2) * cosine

        gap_x = (h / 2) * cosine
        gap_y = (h / 2) * sine

        points = [
            right_x + gap_x, right_y + gap_y,
            left_x + gap_x, left_y + gap_y,
            left_x - gap_x, left_y - gap_y,
            right_x - gap_x, right_y - gap_y
        ]

        return torch.stack(points, dim=-1) + offset  # (N, 8)

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        assert isinstance(scene, np.ndarray)

        height, width = scene.shape[:2]
        scale = max(height, width)  # ou fixe: scale = 500

        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(float)

            # Conversion et normalisation
            xyxy_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
            cxcywh = self.box_xyxy_to_cxcywh(xyxy_tensor)
            cxcywh_normalized = cxcywh / scale

            # G�n�rer les points dans l?espace normalis�
            rbox_pts = self.rbox_cxcywh_to_xyxy(cxcywh_normalized)
            #rbox_pts = self.rbox_cxcywh_to_xyxy(cxcywh)
            # Remise � l?�chelle en pixels
            rbox_pts *= scale

            # Mise en forme pour OpenCV
            rbox_pts = rbox_pts.view(4, 2).numpy().astype(np.int32)
            contour = rbox_pts.reshape((-1, 1, 2))

            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup if custom_color_lookup is None else custom_color_lookup,
            )

            cv2.drawContours(
                image=scene,
                contours=[contour],
                contourIdx=0,
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        return scene
