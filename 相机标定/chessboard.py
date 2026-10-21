import cv2
import numpy as np

def create_chessboard(squares_x=10, squares_y=7, square_size=80, save_path="chessboard.png"):
    row = np.array([0 if i % 2 == 0 else 255 for i in range(squares_x)], dtype=np.uint8)
    board = np.array([row if j % 2 == 0 else 255 - row for j in range(squares_y)], dtype=np.uint8)
    board = np.kron(board, np.ones((square_size, square_size), dtype=np.uint8))

    cv2.imwrite(save_path, board)
    print("saved:", save_path, "size:", board.shape)

    try:
        cv2.imshow("chessboard", board)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("display skipped:", e)

    return board

if __name__ == "__main__":
    create_chessboard()