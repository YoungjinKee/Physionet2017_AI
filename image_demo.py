import tensorflow as tf
import cv2
import time
import argparse
import os

import posenet
import math

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--view', type=str, default='right')

args = parser.parse_args()




def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.25)
            
            
            view=args.view

            #print(pose_scores.shape,keypoint_scores.shape,keypoint_coords.shape)
            if view=='right':
                pose_scores, keypoint_scores, keypoint_coords =pose_scores, keypoint_scores*[0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],keypoint_coords 
            else: 
                pose_scores, keypoint_scores, keypoint_coords =pose_scores, keypoint_scores*[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],keypoint_coords 
            
            #print(pose_scores,keypoint_scores.shape,keypoint_coords.shape)
            keypoint_coords *= output_scale

            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if not args.notxt:
                #print("kkkkkkkkkkkkkkkkkkk")
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
            
            ### Select left or right view ###
            if view=='right':

                keypoint_neck = keypoint_coords[0,4,:]
                keypoint_shoulder = keypoint_coords[0,6,:]
            else:
                keypoint_neck = keypoint_coords[0,5,:]
                keypoint_shoulder = keypoint_coords[0,7,:]
            print(keypoint_neck,keypoint_shoulder)
            
            ### calculate angles ###
            angle = math.atan2(-keypoint_neck[0]+keypoint_shoulder[0], keypoint_neck[1]-keypoint_shoulder[1])
            angle = ((angle*180)/math.pi)
            print("angle = ",90-angle)

            '''
            ### calculate angles ###
            angle = math.atan2(keypoint_neck[0]-keypoint_shoulder[0], keypoint_neck[1]-keypoint_shoulder[1])
            angle = ((angle*180)/math.pi)
            print("angle = ",angle+90)
            '''

        print('Average FPS:', len(filenames) / (time.time() - start))



'''
def get_angle(int x1, int y1, int x2, int y2):
    int dx = x2 - x1;
    int dy = y2 - y1;

    double rad= math.atan2(dx, dy);
    double degree = (rad*180)/math.pi ;

return degree;
'''



if __name__ == "__main__":
    main()
