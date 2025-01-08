import sys

def print_multiindex(p, d):
    idx = [0] * p
    counter = 0
    print(f"{counter}: {idx}")
    counter += 1
    groups = 0
    while True:
        for i in range(p - 1, -1, -1):
            if idx[i] < d - 1:
                idx[i] += 1
                print(f"{counter}: {idx}")
                counter += 1

                for j in range(i + 1, p):
                    idx[j] = idx[j - 1]
                if (idx[-1] + 1) % 8 == 0:
                    print("=======Tile Break=======")
                    groups += 1
                break
        else:
            break
    print("p:", p)
    print("d:", d)
    print("Total number of expanded dim:", counter)
    print("Total number of LDSM groups:", groups * 8)
    print(f"Extra computation: {(groups * 8 - counter) * 100 / counter:.2f}%")


if __name__ == '__main__':
    p, d = int(sys.argv[1]), int(sys.argv[2])
    print_multiindex(p, d)

