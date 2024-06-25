import numpy as np
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import messagebox, filedialog
from tkinter.simpledialog import askinteger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.image import imread

# 出發位置
start_position = (0.5, 0.5)
waypoint_positions = []
num_iterations = 30
num_vehicles = 1
background_image = None
image_aspect_ratio = 1  # 預設比例為1

# 創建主窗口
root = tk.Tk()
root.title("遊戲路線規劃")
root.state('zoomed')  # 最大化窗口

# 創建圖形和畫布
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
ax.set_xticks([])  # 去除X軸標籤
ax.set_yticks([])  # 去除Y軸標籤

# 右側功能區
control_frame = ttk.Frame(root)
control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# 選擇背景圖片
def choose_background():
    global background_image, image_aspect_ratio
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        background_image = imread(file_path)
        img_height, img_width = background_image.shape[:2]
        image_aspect_ratio = img_width / img_height
        adjust_canvas_size()
        refresh_canvas()

# 調整畫布大小
def adjust_canvas_size():
    global fig, ax
    fig.clf()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    fig.set_size_inches(window_width / fig.dpi, window_height / fig.dpi)
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 隱藏坐標軸
    canvas.draw()

# 刷新畫布
def refresh_canvas():
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])  # 去除X軸標籤
    ax.set_yticks([])  # 去除Y軸標籤
    ax.axis('off')  # 隱藏坐標軸
    if background_image is not None:
        ax.imshow(background_image, extent=[0, 1, 0, 1], aspect='auto')
    ax.plot(start_position[0], start_position[1], 'ro')  # 繪製出發點
    for pos in waypoint_positions:
        ax.plot(pos[0], pos[1], 'bo')  # 繪製路點
    canvas.draw()

# 設置迭代次數
def set_iterations():
    global num_iterations
    num_iterations = askinteger("輸入", "請輸入迭代次數:", initialvalue=num_iterations)
    messagebox.showinfo("信息", f"已設置迭代次數為: {num_iterations}")

# 清除路點
def clear_waypoints():
    global waypoint_positions
    waypoint_positions = []
    refresh_canvas()

# 設置位置
def set_position(event):
    global start_position, waypoint_positions
    if event.xdata and event.ydata:
        if event.button == 1:  # 左鍵設置路點
            waypoint_positions.append((event.xdata, event.ydata))
        elif event.button == 3:  # 右鍵設置出發位置
            start_position = (event.xdata, event.ydata)
        refresh_canvas()

# 計算最佳路徑
def calculate_route():
    global distance_matrix, all_routes
    if len(waypoint_positions) == 0:
        messagebox.showwarning("警告", "請添加至少一個路點")
        return

    distance_matrix = generate_distance_matrix(start_position, waypoint_positions)

    num_ants = 50
    alpha = 1.0
    beta = 5.0
    rho = 0.1

    all_routes, best_distance = ant_colony_optimization(distance_matrix, num_vehicles, num_ants, num_iterations, alpha, beta, rho)

    scale.config(to=len(all_routes) - 1)
    plot_route(all_routes[0])

canvas.mpl_connect("button_press_event", set_position)

# 按鈕樣式
button_style = {"bg": "grey", "fg": "black", "font": ("Arial", 12, "normal"), "relief": tk.RAISED, "bd": 3}

# 添加說明標籤
instructions = ttk.Label(control_frame, text="說明:\n1. 左鍵點擊添加路點\n2. 右鍵點擊設置出發點\n3. 使用右側按鈕設置迭代次數\n選擇背景\n清除路點或計算最佳路徑。", font=("Arial", 12, "bold"))
instructions.pack(pady=10)

choose_background_button = tk.Button(control_frame, text="選擇遊戲地圖", command=choose_background, **button_style)
choose_background_button.pack(pady=10)

set_iterations_button = tk.Button(control_frame, text="設置迭代次數", command=set_iterations, **button_style)
set_iterations_button.pack(pady=10)

clear_waypoints_button = tk.Button(control_frame, text="清除路點", command=clear_waypoints, **button_style)
clear_waypoints_button.pack(pady=10)

calculate_route_button = tk.Button(control_frame, text="計算最佳路徑", command=calculate_route, **button_style)
calculate_route_button.pack(pady=10)

# 生成距離矩陣
def generate_distance_matrix(start, waypoints):
    locations = [start] + waypoints
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(i + 1, num_locations):
            distance = np.sqrt((locations[i][0] - locations[j][0]) ** 2 + (locations[i][1] - locations[j][1]) ** 2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

# 螞蟻群體優化算法
def ant_colony_optimization(distance_matrix, num_vehicles, num_ants, num_iterations, alpha, beta, rho):
    num_waypoints = len(distance_matrix) - 1
    pheromone = np.ones(distance_matrix.shape) / len(distance_matrix)
    best_route = None
    best_distance = np.inf
    all_routes = []  # 用於存儲每次迭代的最佳路徑

    def calculate_route_length(route):
        total_distance = 0
        for subroute in route:
            if len(subroute) > 0:
                total_distance += distance_matrix[0, subroute[0]]  # 從出發點到第一個路點
                for i in range(1, len(subroute)):
                    total_distance += distance_matrix[subroute[i - 1], subroute[i]]
                total_distance += distance_matrix[subroute[-1], 0]  # 從最後一個路點回到出發點
        return total_distance

    def select_next_city(probabilities):
        return np.random.choice(range(len(probabilities)), p=probabilities)

    def update_pheromone(pheromone, routes, route_lengths, decay=rho):
        pheromone *= (1 - decay)
        for route, length in zip(routes, route_lengths):
            for subroute in route:
                if len(subroute) > 0:
                    pheromone[0, subroute[0]] += 1.0 / length
                    for i in range(1, len(subroute)):
                        pheromone[subroute[i - 1], subroute[i]] += 1.0 / length
                    pheromone[subroute[-1], 0] += 1.0 / length
        return pheromone

    for iteration in range(num_iterations):
        iteration_routes = []
        all_lengths = []
        for _ in range(num_ants):
            routes = [[] for _ in range(num_vehicles)]
            visited = set()
            for k in range(num_vehicles):
                current_city = 0
                while len(visited) < num_waypoints:
                    probabilities = pheromone[current_city] ** alpha * (np.divide(1.0, distance_matrix[current_city], out=np.zeros_like(distance_matrix[current_city]), where=distance_matrix[current_city] != 0) ** beta)
                    probabilities[list(visited)] = 0  # 確保使用整數列表
                    probabilities /= probabilities.sum()
                    next_city = select_next_city(probabilities)
                    if next_city != 0 and next_city not in visited:
                        routes[k].append(next_city)
                        visited.add(next_city)
                        current_city = next_city
            iteration_routes.append(routes)
            all_lengths.append(calculate_route_length(routes))

        shortest_length = min(all_lengths)
        if shortest_length < best_distance:
            best_distance = shortest_length
            best_route = iteration_routes[np.argmin(all_lengths)]

        pheromone = update_pheromone(pheromone, iteration_routes, all_lengths)
        all_routes.append(best_route)

    return all_routes, best_distance

# 繪製路徑
def plot_route(route):
    refresh_canvas()
    colors = ['b', 'g', 'c', 'm', 'y', 'k']
    for vehicle_id, subroute in enumerate(route):
        color = colors[vehicle_id % len(colors)]
        if len(subroute) > 0:
            start = 0
            for i in subroute:
                ax.plot(
                    [start_position[0] if start == 0 else waypoint_positions[start - 1][0], waypoint_positions[i - 1][0]],
                    [start_position[1] if start == 0 else waypoint_positions[start - 1][1], waypoint_positions[i - 1][1]],
                    color)
                start = i
            ax.plot([waypoint_positions[start - 1][0], start_position[0]],
                    [waypoint_positions[start - 1][1], start_position[1]],
                    color)
    canvas.draw()

# 更新路徑
def on_scale(val):
    iteration = int(val)
    plot_route(all_routes[iteration])

# 添加滑桿來顯示迭代
scale = tk.Scale(control_frame, from_=0, to=30, orient=tk.HORIZONTAL, command=on_scale)
scale.pack(pady=10)

root.mainloop()


