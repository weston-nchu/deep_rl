from flask import Flask, render_template, request, jsonify
import random
import numpy as np

app = Flask(__name__)

discount_factor = 0.9
threshold = 0.01
action_symbols = ['→', '←', '↓', '↑']

def generate_grid(size):
    grid = [['' for _ in range(size)] for _ in range(size)]
    obstacles = set()
    
    while len(obstacles) < size - 2:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if (x, y) not in obstacles:
            obstacles.add((x, y))
            grid[x][y] = 'obstacle'
    
    return grid, obstacles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_grid', methods=['POST'])
def create_grid():
    size = request.json['size']
    grid, obstacles = generate_grid(int(size))
    return jsonify({'grid': grid, 'obstacles': list(obstacles)})

@app.route('/update_start', methods=['POST'])
def update_start():
    return jsonify({'message': 'Start position updated'})

@app.route('/update_goal', methods=['POST'])
def update_goal():
    return jsonify({'message': 'Goal position updated'})

@app.route('/update_dead', methods=['POST'])
def update_dead():
    return jsonify({'message': 'Dead position updated'})

@app.route('/generate_policy', methods=['POST'])
def generate_policy():
    size = int(request.json['size'])
    obstacles = set(tuple(ob) for ob in request.json.get('obstacles', []))
    policy = [[random.choice(action_symbols) if (i, j) not in obstacles else '' for j in range(size)] for i in range(size)]
    return jsonify({'policy': policy})

def value_evaluation(size, obstacles, policy, goal, dead):
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    value_matrix = np.zeros((size, size))
    rewards = np.full((size, size), -0.001)
    
    for x, y in obstacles:
        rewards[x, y] = -1
    rewards[goal[0], goal[1]] = 20
    rewards[dead[0], dead[1]] = -20
    
    while True:
        delta = 0
        new_value_matrix = np.copy(value_matrix)
        for i in range(size):
            for j in range(size):
                if (i, j) in obstacles or (i, j) == goal or (i, j) == dead:
                    continue
                
                dx, dy = actions[action_symbols.index(policy[i][j])]  # Follow given policy
                ni, nj = i + dx, j + dy
                if 0 <= ni < size and 0 <= nj < size:
                    new_value_matrix[i, j] = rewards[ni, nj] + discount_factor * value_matrix[ni, nj]
                else:
                    new_value_matrix[i, j] = rewards[i, j] + discount_factor * value_matrix[i, j]
                
                delta = max(delta, abs(new_value_matrix[i, j] - value_matrix[i, j]))
        
        value_matrix = new_value_matrix
        if delta < threshold:
            break
    
    return value_matrix.tolist()

@app.route('/generate_value_matrix', methods=['POST'])
def generate_value_matrix():
    size = int(request.json['size'])
    obstacles = set(tuple(ob) for ob in request.json.get('obstacles', []))
    goal = tuple(request.json['goal'])
    dead = tuple(request.json['dead'])
    policy = request.json['policy']
    value_matrix = value_evaluation(size, obstacles, policy, goal, dead)
    return jsonify({'value_matrix': value_matrix})

if __name__ == '__main__':
    app.run(debug=True)
