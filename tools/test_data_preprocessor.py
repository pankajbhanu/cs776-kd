from pyexpat import model
from kd.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from torchview import draw_graph

teacher_data_preprocessor = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )

print(teacher_data_preprocessor)


# model_graph = draw_graph(teacher_data_preprocessor, input_size=(3, 640, 640))
# # Save the graph to a file
# model_graph.save('teacher_data_preprocessor.png')
# Display the graph