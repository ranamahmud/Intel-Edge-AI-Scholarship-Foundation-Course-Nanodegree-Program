import argparse
# TODO: Load the necessary libraries
import os
from openvino.inference_engine import IENetwork, IECore
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    # TODO: Load the Inference Engine API
    plugin = IECore()
    # TODO: Load IR files into their related class
    model_binary = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_binary)
    # TODO: Add a CPU extension, if applicable. It's suggested to check
    plugin.add_extension(CPU_EXTENSION, "CPU")
    # your code for unsupported layers for practice before
    # implementing this. Not all of the models may need it.

    # TODO: Get the supported layers of the network

    supported_layers = plugin.query_network(network=net, device_name="CPU")
    # TODO: Check for any unsupported layers, and let the user
    # know if anything is missing. Exit the program, if so.
    unsupported_layers = [layer for layer in net.layers.keys() if layer not in supported_layers]
    # TODO: Load the network into the Inference Engine
    if len(unsupported_layers) != 0:
        print("Unsupported layers found:"+unsupported_layers)
        exit(1)
    plugin.load_network(net, "CPU")
    print("IR successfully loaded into Inference Engine.")

    return


def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()
