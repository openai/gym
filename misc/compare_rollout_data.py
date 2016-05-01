import argparse, numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()
    file1 = np.load(args.file1)
    file2 = np.load(args.file2)

    for k in sorted(file1.keys()):
        arr1 = file1[k]
        arr2 = file2[k]
        if arr1.shape == arr2.shape:
            if np.allclose(file1[k], file2[k]):
                print "%s: matches!"%k
                continue
            else:
                print "%s: arrays are not equal. Difference = %g"%(k, np.abs(arr1 - arr2).max())
        else:
            print "%s: arrays have different shape! %s vs %s"%(k, arr1.shape, arr2.shape)
        print "first 30 els:\n1. %s\n2. %s"%(arr1.flat[:30], arr2.flat[:30])


if __name__ == "__main__":
    main()