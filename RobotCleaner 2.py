# %%
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from queue import PriorityQueue
import numpy as np
import requests
import chardet
from math import sqrt
from flask import Flask, Response
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# %%

# Clase del Agente


class Robot(Agent):
    def __init__(self, unique_id, model, start_pos, papelera_pos):
        super().__init__(unique_id, model)
        self.position = start_pos
        self.capacity = 5
        self.filled = 0
        self.papelera_pos = papelera_pos
        self.flag = True
        self.last_visited_positions = []

    def detectX(self):
        # Step 1: Update robot's knowledge based on neighborhood only can detect X
        neighborhood = self.model.grid.get_neighborhood(
            self.position, moore=True, include_center=False)
        for neighbor in neighborhood:
            row, col = neighbor
            if self.model.grid_data[row][col] == "X":
                self.model.robot_knowledge[row][col] = self.model.grid_data[row][col]
                self.model.obstacles.add((row, col))
        self.model.all_positions = [(row, col) for row, col in self.model.all_positions if (
            row, col) not in self.model.obstacles]

    def detectTrash(self):
        row, col = self.position
        value = self.model.grid_data[row][col]
        if value == "S" or value == "P":
            pass
        else:
            value = int(value)
            self.model.robot_knowledge[row][col] = value
            if value > 0:
                self.model.trashes.add((row, col))

    def collect_trash(self):
        row, col = self.position
        value = self.model.grid_data[row][col]

        if self.filled < self.capacity:
            trash = int(value)
            remaining_capacity = self.capacity - self.filled
            if trash >= remaining_capacity:
                self.filled = self.capacity
                self.model.robot_knowledge[row][col] = str(
                    trash - remaining_capacity)
                self.model.grid_data[row][col] = str(
                    trash - remaining_capacity)
                # print(f"Agent {self.unique_id} collected {remaining_capacity} trash. Total collected: {self.filled}")
                # self.model.trashes.add((row, col))
            else:
                self.filled += trash
                self.model.robot_knowledge[row][col] = "0"
                self.model.grid_data[row][col] = "0"
                self.model.trashes.remove(self.position)
                # print(f"Agent {self.unique_id} collected {trash} trash. Total collected: {self.filled}")

    def calculate_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)

    def stepsCloser(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        new_x = pos1[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
        new_y = pos1[1] + (1 if dy > 0 else -1 if dy < 0 else 0)

        return (new_x, new_y)

    def moveExplore(self):
        # print("ID:" , self.unique_id, "POS", self.pos, "Cant: ", self.filled)
        self.detectX()
        self.detectTrash()

        neighborhood = self.model.grid.get_neighborhood(
            self.position, moore=True, include_center=False)

        # Crear listas para almacenar la atracción y las posiciones disponibles
        attractiveness = []
        available_positions = set()

        for neighbor in neighborhood:
            row, col = neighbor
            neighbor = (row, col)  # Convertir las coordenadas a una tupla
            if (self.model.robot_knowledge[row][col] == ".") and neighbor not in self.model.occupied_positions:
                # Evitar las celdas con "X"
                if self.model.grid_data[row][col] != "X" and self.model.robot_knowledge[row][col] not in self.model.visited_positions:
                    attraction = 1  # Atracción base
                    available_positions.add(neighbor)

            else:
                attractiveness.append(0)

        available_positions = [
            pos for pos in available_positions if pos not in self.last_visited_positions]

        if any(available_positions):
            weights = [
                1 if pos in available_positions else 0 for pos in neighborhood]
            new_position = self.random.choices(
                neighborhood, weights=weights)[0]
        else:
            # print("ID: ", self.unique_id)
            # Si no hay posiciones disponibles, intentar moverse a una posición aleatoria no ocupada
            if self.model.all_positions:
                closest_unvisited = min(
                    self.model.all_positions, key=lambda pos: self.calculate_distance(self.position, pos))
                next_pos = self.stepsCloser(self.position, closest_unvisited)
                # print("Pos, closesr, nextmove: ",self.position, closest_unvisited, next_pos)

                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                        and pos not in self.last_visited_positions
                    ]

                    if valid_neighbors:
                        new_position = min(valid_neighbors, key=lambda pos: self.calculate_distance(
                            pos, closest_unvisited))

                        # print("New pos de diagonal: ", new_position)
                    else:
                        new_position = self.position
                        # print("Stay no posible moves")
                else:
                    new_position = next_pos
                    # print("Next move: ", next_pos)
            else:
                new_position = self.position
                # print("No moves filled")

        # Convertir la nueva posición en una tupla
        new_position = tuple(new_position)
        self.model.visited_positions.add(new_position)

        if new_position in self.model.all_positions:
            self.model.all_positions.remove(new_position)

        if self.position in self.model.occupied_positions:
            self.model.occupied_positions.remove(self.position)
        self.model.occupied_positions.add(new_position)

        self.model.grid.move_agent(self, new_position)
        self.position = new_position
        if len(self.last_visited_positions) >= 5:
            self.last_visited_positions.pop(0)
        self.last_visited_positions.append(new_position)

    # Funcion de recojer basura

    def moveToTrash(self):
        print(len(self.model.trashes))
        # print("ID:" , self.unique_id, "POS", self.pos, "Cant: ", self.filled, "len: ", len(self.model.trashes) )
        if self.flag:
            if self.position not in self.model.visited_positions:
                self.model.visited_positions.add(self.position)
            row, col = self.position
            value = self.model.grid_data[row][col]
            self.model.robot_knowledge[row][col] = value
            if value != "0" and value != "P" and value != "R" and value != "X":
                self.model.trashes.add(self.position)
                self.flag = False
            self.flag = False

        if self.position in self.model.trashes:
            self.collect_trash()
            # print("remove: ", new_position)

        if self.position == self.papelera_pos:
            self.filled = 0

        # if capacid----------
        # print("ID:" , self.unique_id, "POS", self.pos, "Cant: ", self.filled, "len: ", len(self.model.trashes) )

        if self.filled == self.capacity or not self.model.trashes:
            if self.filled == 0:
                corner_position = (0, 0)

                next_pos = self.stepsCloser(self.position,  corner_position)
                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                    ]

                    if valid_neighbors:
                        new_position = min(
                            valid_neighbors, key=lambda pos: self.calculate_distance(pos, corner_position))
                        # print("New pos de diagonal: ", new_position)
                    else:
                        new_position = self.position
                        # print("Stay no posible moves")
                else:
                    new_position = next_pos
                    # print("Next move: ", next_pos)
            else:
                next_pos = self.stepsCloser(self.position,  self.papelera_pos)
                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                        and pos not in self.last_visited_positions
                    ]

                    if valid_neighbors:
                        new_position = min(valid_neighbors, key=lambda pos: self.calculate_distance(
                            pos, self.papelera_pos))
                        # print("New pos de diagonal: ", new_position)
                    else:
                        new_position = self.position
                        # print("Stay no posible moves")
                else:
                    new_position = next_pos
                    # print("Next move: ", next_pos)

        else:
            if self.model.trashes:
                closest_unvisited = min(
                    self.model.trashes, key=lambda pos: self.calculate_distance(self.position, pos))
                next_pos = self.stepsCloser(self.position, closest_unvisited)
                # print("Pos, closesr, nextmove: ",self.position, closest_unvisited, next_pos)

                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                        and pos not in self.last_visited_positions
                    ]

                    if valid_neighbors:
                        new_position = min(valid_neighbors, key=lambda pos: self.calculate_distance(
                            pos, closest_unvisited))

                        # print("New pos de diagonal: ", new_position)
                    else:
                        new_position = self.position
                        # print("Stay no posible moves")
                else:
                    new_position = next_pos
                    # print("Next move: ", next_pos)
            else:
                new_position = self.position
                # print("No moves filled")

        # print("ID: ", self.unique_id, "   --- Basura: ", self.filled, "   ---Pos: ", self.position, "   ---Next pos: ", new_position)
        new_position = tuple(new_position)

        if self.position in self.model.occupied_positions:
            self.model.occupied_positions.remove(self.position)
        self.model.occupied_positions.add(new_position)

        self.model.grid.move_agent(self, new_position)
        self.position = new_position

        if len(self.model.trashes) != 0:
            if len(self.last_visited_positions) >= 5:
                self.last_visited_positions.pop(0)
            self.last_visited_positions.append(new_position)


# %%

# Modelo
class OfficeCleaningModel(Model):
    def __init__(self, grid_data, height, width):

        self.grid_data = grid_data
        self.height = height
        self.width = width
        self.robot_knowledge = [["R" if value == "S" else "P" if value ==
                                 "P" else "." for value in row] for row in self.grid_data]
        self.all_positions = [(row, col) for row in range(
            self.height) for col in range(self.width)]
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.schedule = RandomActivation(self)
        self.agent_count = 0

        self.occupied_positions = set()
        self.visited_positions = set()
        self.obstacles = set()
        self.trashes = set()
        self.unallowed_positions = set()

        self.create_agents()
        self.papelera_pos = self.find_papelera()

        self.exploring = True
        self.mision_complete = False

        self.datacollector = DataCollector(
            agent_reporters={}
        )

    def find_papelera(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.grid_data[row][col] == "P":
                    papelera_pos = (row, col)
                    if papelera_pos in self.all_positions:
                        self.all_positions.remove(papelera_pos)
                    return papelera_pos

    def create_agents(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.grid_data[row][col] == "S":
                    self.all_positions.remove((row, col))
                    self.visited_positions.add((row, col))
                    for i in range(5):
                        robot = Robot(i, self, (row, col),
                                      self.find_papelera())
                        self.schedule.add(robot)
                        self.grid.place_agent(robot, (row, col))

    # La funcion print_grid, se encarga de impirmir las diferentes matrices, en este caso es la matriz que se envia a Unity.
    def print_grid(self):
        with open("input.txt", "w") as file:
            for row_idx, row in enumerate(self.grid_data):
                for col_idx, cell_value in enumerate(row):
                    position = (row_idx, col_idx)
                    cell_contents = self.grid.get_cell_list_contents(position)
                    if cell_contents:
                        agent_count = len(cell_contents)
                        if agent_count > 1:
                            file.write("A ")
                        else:
                            agent_ids = [str(agent.unique_id)
                                         for agent in cell_contents]
                            agent_ids_str = " ".join(agent_ids)
                            file.write(f"A{agent_ids_str} ")
                    else:
                        file.write(f"{cell_value} ")
                file.write("\n")

    # La funcion print_grid_withAgents se encarga de imprimir la matriz con los agentes, se uso en el desarrollo, para tener una nocion de los
    # agentes y sus posiciones.
    def print_grid_withAgents(self):
        self.exploringmap = []
        for row in range(self.grid.width):
            for col in range(self.grid.height):
                position = (row, col)
                cell_contents = self.grid.get_cell_list_contents(position)
                if cell_contents:
                    agent_count = len(cell_contents)
                    print(f"{agent_count}", end=" ")
                else:
                    if position in self.visited_positions:
                        print("#", end=" ")
                    elif position in self.obstacles:
                        print("X", end=" ")
                    elif position == self.find_papelera():
                        print("P", end=" ")
                    else:
                        print(".", end=" ")
            print()

    # La funcion step, se encarga de mandar a llamar todas las fucniones, actualizacion de agentes, registro de datos, etc...
    def step(self):
        if len(self.all_positions) != 0:
            for agent in self.schedule.agents:
                agent.moveExplore()
        else:
            self.exploring = False

            if len(model.trashes) != 0:
                for agent in self.schedule.agents:
                    agent.moveToTrash()
            else:
                if all(agent.filled == 0 for agent in self.schedule.agents):
                    self.mision_complete = True
                else:
                    for agent in self.schedule.agents:
                        agent.moveToTrash()

        self.print_grid()
        self.datacollector.collect(self)
        self.schedule.step()

# %%
# Abrimos el archivo


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
        return result['encoding']


# Variable con el nombre de nuestro archivo a analizar
mapa = "input1.txt"
file_encoding = detect_encoding(mapa)

with open(mapa, "r", encoding=file_encoding) as file:
    first_line = file.readline().strip()

# Extraemos la primera linea con el width y height
height, width = map(int, first_line.split())

# Leemos el resto de archivo
with open(mapa, "r", encoding=file_encoding) as file:
    lines = file.readlines()[1:]  # Ignore the first line
    grid_data = [line.strip().split() for line in lines]

# %%
app = Flask(__name__)
model = OfficeCleaningModel(grid_data, height, width)


def read_matrix_from_file(filename):
    matrix_data = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split()
            matrix_data.append(row)
    return matrix_data


@app.route('/get_matrix')
def publish_matrix():
    matrix = read_matrix_from_file('input.txt')
    matrix_str = "\n".join([" ".join(row) for row in matrix])

    return Response(matrix_str, content_type='text/plain')


@app.route('/step')
def run_step():
    if not model.mision_complete:
        model.step()
        return "", 200
    else:
        return "", 400


if __name__ == '__main__':
    app.run()


# %%
