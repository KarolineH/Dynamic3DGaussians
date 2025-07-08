
import torch
import torchvision
import numpy as np
import pathlib
import json
import cv2
from helpers import setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import os
import shutil

class OutputRenderer():

    '''
    Class for rendering the results of the Dynamic 3D Gaussians model.
    This only generates the colour and depth renders for the test cameras and finds the corresponding ground truth images.
    It also combines the colour renders into mp4 videos.
    '''

    dataset_location = '/workspace/synthetic_data/'

    def __init__(self, w=800, h=800, near=0.01, far=100.0):
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.frame_rate = 24

    def render_tests_to_file(self, exp, seq, cam_id=None, copy_gt=False):

        '''Renders colour images and depth images for the test cameras to png files.'''

        # first fetch the camera parameters and the trained model
        ids, extrinsics, intrinsics = self.get_test_cameras(seq) # Camera parameters for all of the test cameras
        scene_data, is_fg = self.load_scene_data(seq, exp)

        # Create the output directories
        render_dir = f"/workspace/Dynamic3DGaussians/output/{exp}/{seq}/renders/"
        gt_dir = f"/workspace/Dynamic3DGaussians/output/{exp}/{seq}/gt/"
        video_dir = f"/workspace/Dynamic3DGaussians/output/{exp}/{seq}/colour_videos/"
        depth_dir = f"{pathlib.Path(__file__).parent}/output/{exp}/{seq}/depth_renders/"
        for path in [render_dir, gt_dir, video_dir, depth_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Now loop over frames
        j=0
        for t in range(len(scene_data)):
            for i,view in enumerate(ids): # Iterate over all test cameras, unless a specific camera is given
                if cam_id is not None:
                    if view != cam_id:
                        continue

                w2c = extrinsics[i]
                k = intrinsics[i]

                # RENDER
                im, depth = self.render(w2c, k, scene_data[t])
                
                # For displaying you could convert them:
                # to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
                # to8b(im).transpose(1,2,0)
                torchvision.utils.save_image(im, os.path.join(render_dir, f"{j:05d}.png"))

                max_valid_depth = torch.unique(depth)[-2] # This should be the maximum reading that is not 15.0, which is the predefined far max value
                min_depth = torch.min(depth)
                norm = ((depth - min_depth) / (max_valid_depth - min_depth)) * 0.8
                norm = torch.clip(norm, 0, 1)
                torchvision.utils.save_image(norm, os.path.join(depth_dir, f"{j:05d}.png"))

                if copy_gt:
                    shutil.copyfile(os.path.join(self.dataset_location, seq, 'ims', str(view), f"{t+1:06d}.png"), os.path.join(gt_dir, f"{j:05d}.png"))

                j += 1
        
    def render(self, w2c, k, timestep_data):
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, k, w2c, self.near, self.far)
            im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth
    
    def get_test_cameras(self, seq):
        ''' Find the meta data for the cameras in the test batch.'''
        test_md_path = os.path.join(self.dataset_location, seq, "test_meta.json")
        md = json.load(open(test_md_path))
        camera_ids = np.array(md['cam_id'][0]) # IDs of the test cameras
        w2cs = np.array(md['w2c'])[0] # world to camera matrices (n, 4, 4)
        ks = np.array(md['k'])[0] # intrinsic matrices (n, 3, 3)
        return camera_ids, w2cs, ks
        
    def load_scene_data(self, seq, exp, seg_as_col=False):
        params = dict(np.load(f"/workspace/Dynamic3DGaussians/output/{exp}/{seq}/params.npz"))
        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        for t in range(len(params['means3D'])):
            rendervar = {
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            # if REMOVE_BACKGROUND:
            #     rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)
        # if REMOVE_BACKGROUND:
        #     is_fg = is_fg[is_fg]
        return scene_data, is_fg
    

    def renders_to_mp4(self, exp, seq, out_dir_name='colour_videos'):
        '''
        Fetches the rendered colour images (test set output) and stitches them together into mp4 videos.
        Assumes 4 test views per model and that images are interleaved.
        '''
        sequence_path = f"/workspace/Dynamic3DGaussians/output/{exp}/{seq}/"
        out_path = os.path.join(sequence_path, out_dir_name)
        output_template = "view_{}.mp4"
        render_dir = os.path.join(sequence_path, "renders") 
        if not os.path.exists(render_dir):
            print(f"Render directory {render_dir} does not exist. Skipping video creation.")
            return
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        images = sorted([f for f in os.listdir(render_dir) if f.endswith(".png")])

        # Create video writers
        # One for each of the four test views, because images are interleaved
        writers = [
            cv2.VideoWriter(os.path.join(out_path, output_template.format(i)), cv2.VideoWriter_fourcc(*'mp4v'), self.frame_rate, (self.w, self.h))
            for i in range(4)
        ]

        # Write frames to appropriate video
        for j, img_name in enumerate(images):
            frame = cv2.imread(os.path.join(render_dir, img_name))
            writers[j % 4].write(frame)

        # Release writers
        import subprocess
        for i,writer in enumerate(writers):
            writer.release()
            outfile = os.path.join(out_path,output_template.format(i))
            fix_video_codec(outfile)  # Ensure video codec is correct

def fix_video_codec(outpath):
    import subprocess
    temp_path = outpath + '.tmp.mp4'
    cmd = [
        'ffmpeg', '-i', outpath,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-movflags', '+faststart',
        temp_path,
        '-y'
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(temp_path, outpath)  # atomically overwrite original
        print(f"✔ Converted: {outpath}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed on {outpath}:", e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return

if __name__ == "__main__":

    OR = OutputRenderer()
    exp_name = "d3_exp06"
    for sequence in ["ani_growth", "bending", "branching", "colour", "hole", "large_growth", "rotation", "shedding", "stretching", "translation", "twisting", "uni_growth"]:
        OR.render_tests_to_file(exp_name, sequence, copy_gt=True)
        OR.renders_to_mp4(exp_name, sequence)
        print(f"Rendered and saved outputs for {sequence} in {exp_name}.")