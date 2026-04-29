from __future__ import annotations


def main() -> None:
    name = "OpenCV Demo"
    values = [1, 2, 3, 4, 5]
    total = sum(values)
    average = total / len(values)

    print(f"Project: {name}")
    print(f"Values: {values}")
    print(f"Total: {total}")
    print(f"Average: {average:.2f}")


if __name__ == "__main__":
    main()
