import pygame
from collections import deque
from SnakeGameAI import SnakeGameAI, Direction, Point, BLOCK_SIZE

class AgentBFS:
    def bfs(self, start, goal, obstacles, grid_width, grid_height):
        queue = deque()
        queue.append(start)#başlangıç noktasını queueya ekledik
        came_from = {}# her noktanın geldiği noktayı tuttu
        visited = set()#ziyaret edilen yerleri tuttu
        visited.add(start)#başlangıç ziyaret edildi

        while queue:
            current = queue.popleft() #queuedaki ilk elemanı aldık
            if current == goal:# hedefe ulaştıysak
                path = [current]#path oluştur
                while current in came_from:
                    current = came_from[current]#geriye doğru git
                    path.append(current)
                path.reverse()# bu şekilde baştan sona sıraladık
                return path

            #4 farklı komşuya bakıyor
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:#sağ sol yukarı aşağı neighborlar
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < grid_width and
                    0 <= neighbor[1] < grid_height and
                    neighbor not in obstacles and
                    neighbor not in visited):#sınırların içinde mi, engele geldi mi ve ziyaret edildi mi kontrolü yaprık
                    
                    visited.add(neighbor)
                    came_from[neighbor] = current# bu noktaya nasıl geldik kaydet
                    queue.append(neighbor)
        return None

    #yeme ulaşma hamleleri
    def get_action(self, game: SnakeGameAI):
        grid_width = game.w // BLOCK_SIZE
        grid_height = game.h // BLOCK_SIZE

        start = (int(game.head.x // BLOCK_SIZE), int(game.head.y // BLOCK_SIZE))
        goal = (int(game.food.x // BLOCK_SIZE), int(game.food.y // BLOCK_SIZE))

        obstacles = set()
        for pt in game.snake[1:]:
            obstacles.add((int(pt.x // BLOCK_SIZE), int(pt.y // BLOCK_SIZE)))

        path = self.bfs(start, goal, obstacles, grid_width, grid_height)

        if path is None or len(path) < 2:
            return [1, 0, 0] # yol bulunamazsa düz git

        # yoldaki bir sonraki hücreyi aldık
        next_cell = path[1]
        desired_direction = None

        # cell'e göre istenen yönü belirledik
        if next_cell[0] > start[0]:#x koordinatından büyükse sağa git
            desired_direction = Direction.RIGHT
        elif next_cell[0] < start[0]:#x koordinatından küçükse sola git
            desired_direction = Direction.LEFT
        elif next_cell[1] > start[1]:##y koordinatından büyükse aşağı git
            desired_direction = Direction.DOWN
        elif next_cell[1] < start[1]:##y koordinatından küçükse yukarı git
            desired_direction = Direction.UP

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clock_wise.index(game.direction)
        desired_idx = clock_wise.index(desired_direction)

        #yılanın nereye gitmesi gerektiğini belirledik
        if desired_direction == game.direction:#mevcut yön ile istenen yön aynı ise
            return [1, 0, 0]#düz
        elif (current_idx + 1) % 4 == desired_idx:#hedef yön mevcut yönün saat yönünde  bir sonraki yönü ise 
            return [0, 1, 0]#sağa dön
        elif (current_idx - 1) % 4 == desired_idx:#hedef yön mevcut yönün saat yönünün tersine bir önceki yönü ise
            return [0, 0, 1]#sola dön
        else:
            return [1, 0, 0]#düz ama istenmeyen durum

if __name__ == '__main__':
    game = SnakeGameAI()
    agent = AgentBFS()
    while True:
        action = agent.get_action(game)
        reward, game_over, score = game.play_step(action)
        if game_over:
            print("Oyun bitti! Skor:", score)
            game.reset()
