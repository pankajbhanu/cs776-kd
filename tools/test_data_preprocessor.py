from kd.models.data_preprocessors.data_preprocessor import DetDataPreprocessor


teacher_data_preprocessor = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )

print(teacher_data_preprocessor)