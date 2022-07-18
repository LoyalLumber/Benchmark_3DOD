import resource_usage

def main():
    
    parser = argparse.ArgumentParser(description="Run resource analyzer.")
    parser.add_argument("-m", "--model", type=str, default="Complex-YOLOv3", help="Choose object detectors (check the available list)" (.cfg)")
    parser.add_argument("-d", "--datasets", type=str, default="KITTI", help="KITTI or nuScenes (check the compatibility of each detector)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    analyze_resource(args)

if __name__ == '__main__':
    try:ddd
        main()
    except: pass