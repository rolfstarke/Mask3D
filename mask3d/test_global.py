from mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh
import torch
import glob
import os

model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Find all .ply files in the data folder
ply_files = glob.glob('data/*.ply')

for pointcloud_file in ply_files:
    # Load input data
    mesh = load_mesh(pointcloud_file)

    # Prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

    # Run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    
    # Map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)

    # Construct the output filename by appending _labelled before the .ply extension
    base_name = os.path.basename(pointcloud_file)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join('data', f"{file_name_without_ext}_labelled.ply")

    # Save colorized mesh
    save_colorized_mesh(mesh, labels, output_file_path, colormap='scannet200')
