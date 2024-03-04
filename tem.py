import tensorflow as tf

# Path to frozen graph file
frozen_graph_path = 'frozen_inference_graph.pb'

# Load frozen graph
with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Convert graph to text format
text_format = tf.compat.v1.GraphDef().SerializeToString()
tf.io.write_graph(graph_def, './', 'frozen_graph.pbtxt', as_text=True)
