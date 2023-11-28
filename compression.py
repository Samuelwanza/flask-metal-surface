import h5py
import gzip

# Path to the original .h5 file
original_file_path = 'metals_surface_defects_detection_model.h5'

# Path to the compressed .h5 file
compressed_file_path = 'model_compressed.h5'

# Save the model with gzip compression
with h5py.File(original_file_path, 'r') as f_in:
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(f_in['model_weights'].value.tobytes())
