import math
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from matplotlib import pyplot as plt


def main():
    root = tkinter.Tk()
    root.title("test mpl")
    # 描画領域
    fig = plt.figure(figsize=(10, 6))
    # 描画するデータ
    x = np.arange(0, 10, 0.1)
    y = [math.sin(i) for i in x]

    # グラフを描画する
    plt.plot(x, y) 

    #グラフ表示
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().pack()

    toolbar=NavigationToolbar2Tk(canvas, root)
    root.mainloop()


if __name__ == '__main__':
    main()