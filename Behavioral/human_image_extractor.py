"""Node to extract images and/or video from Spot and pass it to a facial recognition node"""

# Matteo Corrado - 29/10/2025

import sys
import argparse

import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

import argparse
import math
import sys
import time

import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import (basic_command_pb2, geometry_pb2, image_pb2, manipulation_api_pb2,
                        network_compute_bridge_pb2)
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.network_compute_bridge_client import (ExternalServerError,
                                                         NetworkComputeBridgeClient)
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, block_until_arm_arrives)
from bosdyn.client.robot_state import RobotStateClient

# List of image sources from 'fetch.py'
kImageSources = [
    'frontleft_fisheye_image', 'frontright_fisheye_image', 'left_fisheye_image',
    'right_fisheye_image', 'back_fisheye_image'
]

def create_robot_connection(options):
    """Create and authenticate a robot connection."""
    sdk = bosdyn.client.create_standard_sdk('image_extractor')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(), "Robot is estopped. Please clear the estop and try again."
    return robot

def get_obj_and_img(network_compute_client, server, model, confidence, image_sources, label):

    for source in image_sources:
        # Build a network compute request for this image source.
        image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
            image_source=source)

        # Input data:
        #   model name
        #   minimum confidence (between 0 and 1)
        #   if we should automatically rotate the image
        input_data = network_compute_bridge_pb2.NetworkComputeInputData(
            image_source_and_service=image_source_and_service, model_name=model,
            min_confidence=confidence, rotate_image=network_compute_bridge_pb2.
            NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL)

        # Server data: the service name
        server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
            service_name=server)

        # Pack and send the request.
        process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
            input_data=input_data, server_config=server_data)

        try:
            resp = network_compute_client.network_compute_bridge_command(process_img_req)
        except ExternalServerError:
            # This sometimes happens if the NCB is unreachable due to intermittent wifi failures.
            print('Error connecting to network compute bridge. This may be temporary.')
            return None, None, None

        best_obj = None
        highest_conf = 0.0
        best_vision_tform_obj = None

        img = get_bounding_box_image(resp)
        image_full = resp.image_response

        # Show the image
        cv2.imshow("Fetch", img)
        cv2.waitKey(15)

        if len(resp.object_in_image) > 0:
            for obj in resp.object_in_image:
                # Get the label
                obj_label = obj.name.split('_label_')[-1]
                if obj_label != label:
                    continue
                conf_msg = wrappers_pb2.FloatValue()
                obj.additional_properties.Unpack(conf_msg)
                conf = conf_msg.value

                try:
                    vision_tform_obj = frame_helpers.get_a_tform_b(
                        obj.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                        obj.image_properties.frame_name_image_coordinates)
                except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                    # No depth data available.
                    vision_tform_obj = None

                if conf > highest_conf and vision_tform_obj is not None:
                    highest_conf = conf
                    best_obj = obj
                    best_vision_tform_obj = vision_tform_obj

        if best_obj is not None:
            return best_obj, image_full, best_vision_tform_obj

    return None, None, None

def main(argv):
    """Main function to extract images from Spot robot and pass them to a model that recognises humans, emulating Boston Dynamic's fetch example."""
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help='Time [ms] to wait before the next capture',
                        type=int, default=100)
    parser.add_argument('-r', '--resize-ratio', help='Fraction to resize the image', type=float,
                        default=1)
    parser.add_argument(
        '--disable-full-screen',
        help='A single image source gets displayed full screen by default. This flag disables that.',
        action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)

    # Create robot object with an image client.
    robot = create_robot_connection(options)
    image_client = robot.ensure_client(options.image_service)
    requests = [
        build_image_request(source, quality_percent=options.jpeg_quality_percent,
                            resize_ratio=options.resize_ratio) for source in options.image_sources
    ]

    for image_source in options.image_sources:
        print(f"Extracting images from source: {image_source}")

    # Further processing would go here, such as passing images to a facial recognition model.