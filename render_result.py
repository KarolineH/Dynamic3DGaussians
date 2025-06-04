
import torch
import numpy as np
import pathlib
import json
import imageio
import cv2
from PIL import Image
from helpers import setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import os

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

    def render_tests_to_file(self, exp, seq, cam_id=None):

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
                
                # Save the colour render
                image = (im.cpu().permute(1,2,0).contiguous().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(render_dir, f"{j:05d}.png"), image)

                # find and copy the correct ground truth image 
                gt_im = os.path.join(self.dataset_location, seq, 'ims', str(view), f"{t+1:06d}.png")
                if os.path.exists(gt_im):
                    gt_with_bg = self.gt_insert_bg(gt_im, bg=0)
                    gt_with_bg.save(os.path.join(gt_dir, f"{j:05d}.png"))
                else:
                    print(f"Ground truth image {gt_im} does not exist, skipping.")
                            
                inv_depth_array = depth.cpu().permute(1,2,0).contiguous().numpy() # depth images come out inverted, with far values being smaller than near values
                depth_array = np.max(inv_depth_array) - inv_depth_array # now the depth values are in the range [0, max_depth]
                norm_depth_array = depth_array / np.max(depth_array) * 255.0 # normalise to [0, 255] range
                depth_im = norm_depth_array.astype(np.uint8).squeeze()
                imageio.imwrite(os.path.join(depth_dir, f"{j:05d}.png"), depth_im)
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
    
    def gt_insert_bg(self, im_file, bg=0):
        '''
        Insert a white or black background into an original ground truth image
        that had a transparent background (alpha channel).
        bg = 0 is black and 1 is white
        '''

        img = Image.open(im_file).convert("RGBA")
        if bg == 0:
            bg_colour = Image.new("RGBA", img.size, (0, 0, 0, 255))
        elif bg == 1:
            bg_colour = Image.new("RGBA", img.size, (255, 255, 255, 255))
            
        # Composite the original image over the black background
        composited = Image.alpha_composite(bg_colour, img)
        final = composited.convert("RGB")
        return final
    

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
        for writer in writers:
            writer.release()


if __name__ == "__main__":

    OR = OutputRenderer()
    exp_name = "dynamic_gaussians_exp03"
    for sequence in ["ani_growth", "bending", "branching", "colour", "hole", "rotation", "shedding", "stretching", "translation", "twisting", "uni_growth"]:
        OR.render_tests_to_file(exp_name, sequence)
        OR.renders_to_mp4(exp_name, sequence)