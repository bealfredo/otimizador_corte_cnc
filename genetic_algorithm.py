from common.layout_display import LayoutDisplayMixin
import torch
import matplotlib.pyplot as plt


class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa. Executado por Marco.")
        self.TAM_POP = TAM_POP
        self.initial_layout = recortes_disponiveis  # Available cut parts
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []
        self.POP_AUX = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.initialize_population()
        self.melhor_aptidoes = []
        self.optimized_layout = None  # To be set after optimization

        # Configura dispositivo (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matriz_chapa = torch.zeros((sheet_height, sheet_width), device=self.device, dtype=torch.bool)


    def initialize_population(self):
        # Initialize the population of individuals.
        pass

    def evaluate(self):
        # Evaluate the fitness of individuals based on available parts.
        pass

    def genetic_operators(self):
        # Execute genetic operators (crossover, mutation, etc.) to evolve the population.
        pass

    def run(self):
        # Main loop of the evolutionary algorithm.

        auxCanvas = AuxCanvas(self.sheet_height, self.sheet_width, self.sheet_width, self.sheet_height, self.matriz_chapa)
        auxCanvas.desenhar_peca_aleatoria(self.initial_layout)
        # Visualizar o resultado
        canvas_np = self.matriz_chapa.cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(canvas_np, cmap='gray')
        plt.title("Canvas com Peças em Posições Aleatórias")
        plt.axis('on')
        plt.grid(True, alpha=0.3)
        plt.show()

        
        

        # Temporary return statement to avoid errors
        self.optimized_layout = self.initial_layout
        return self.optimized_layout

    def optimize_and_display(self):
        """
        Displays the initial layout, runs the optimization algorithm,
        and displays the optimized layout using the mixin's display_layout method.
        """
        # Display initial layout
        self.display_layout(self.initial_layout, title="Initial Layout - Genetic Algorithm")
        
        # Run the optimization algorithm (updates self.melhor_individuo)
        self.optimized_layout = self.run()
        
        # Display optimized layout
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm")
        return self.optimized_layout
    

class AuxCanvas():
    def __init__(self, altura_canvas, largura_canvas, sheet_width, sheet_height, canvas):
        self.altura_canvas = altura_canvas
        self.largura_canvas = largura_canvas
        self.canvas = canvas
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height

    # Funções para desenhar formas
    def desenhar_retangulo(canvas, x, y, largura, altura):
        x, y = int(x), int(y)
        largura, altura = int(largura), int(altura)
        x_end = min(x + largura, canvas.shape[1])
        y_end = min(y + altura, canvas.shape[0])
        if x >= 0 and y >= 0 and x_end > x and y_end > y:
            canvas[y:y_end, x:x_end] = 1

    def desenhar_diamante(canvas, x, y, largura, altura):
        x, y = int(x), int(y)
        largura, altura = int(largura), int(altura)
        metade_largura = largura // 2
        metade_altura = altura // 2
        centro_x = x + metade_largura
        centro_y = y + metade_altura
        for i in range(max(0, y), min(canvas.shape[0], y + altura)):
            for j in range(max(0, x), min(canvas.shape[1], x + largura)):
                if (abs(j - centro_x) / metade_largura + abs(i - centro_y) / metade_altura) <= 1:
                    canvas[i, j] = 1

    def desenhar_circulo(canvas, x, y, raio):
        x, y = int(x), int(y)
        raio = int(raio)
        for i in range(max(0, y - raio), min(canvas.shape[0], y + raio + 1)):
            for j in range(max(0, x - raio), min(canvas.shape[1], x + raio + 1)):
                if (j - x)**2 + (i - y)**2 <= raio**2:
                    canvas[i, j] = 1

    # Função para verificar sobreposição
    def tem_sobreposicao(canvas, x, y, largura, altura, tipo, raio=None):
        if tipo == "retangular":
            x_end = min(x + largura, canvas.shape[1])
            y_end = min(y + altura, canvas.shape[0])
            if x < 0 or y < 0 or x_end <= x or y_end <= y:
                return True
            region = canvas[y:y_end, x:x_end]
            return torch.any(region == 1)
        
        elif tipo == "diamante":
            metade_largura = largura // 2
            metade_altura = altura // 2
            centro_x = x + metade_largura
            centro_y = y + metade_altura
            for i in range(max(0, y), min(canvas.shape[0], y + altura)):
                for j in range(max(0, x), min(canvas.shape[1], x + largura)):
                    if (abs(j - centro_x) / metade_largura + abs(i - centro_y) / metade_altura) <= 1:
                        if canvas[i, j] == 1:
                            return True
            return False
        
        elif tipo == "circular":
            # Verificar se o círculo está completamente dentro do canvas
            if x - raio < 0 or y - raio < 0 or x + raio >= canvas.shape[1] or y + raio >= canvas.shape[0]:
                return True
            
            for i in range(max(0, y - raio), min(canvas.shape[0], y + raio + 1)):
                for j in range(max(0, x - raio), min(canvas.shape[1], x + raio + 1)):
                    if (j - x)**2 + (i - y)**2 <= raio**2:
                        if canvas[i, j] == 1:
                            return True
            return False
        
        
    def desenhar_peca_aleatoria(self, recortes_disponiveis):
        
        # Desenhar peças em posições aleatórias
        for indice, recorte in enumerate(recortes_disponiveis):
            tipo = recorte["tipo"]
            
            # Definir dimensões baseadas no tipo da peça
            if tipo == "retangular":
                largura = recorte["largura"]
                altura = recorte["altura"]
                x_max = self.largura_canvas - largura
                y_max = self.altura_canvas - altura
            elif tipo == "diamante":
                largura = recorte["largura"]
                altura = recorte["altura"]
                x_max = self.largura_canvas - largura
                y_max = self.altura_canvas - altura
            elif tipo == "circular":
                raio = recorte["r"]
                x_max = self.largura_canvas - 2 * raio
                y_max = self.altura_canvas - 2 * raio
            
            # Garantir valores não negativos
            x_max = max(1, x_max)
            y_max = max(1, y_max)
            
            # Gerar posição aleatória
            x = torch.randint(0, x_max, (1,), device='cuda').item()
            y = torch.randint(0, y_max, (1,), device='cuda').item()
            
            # Verificar sobreposição
            sobreposicao = False
            if tipo == "retangular":
                sobreposicao = AuxCanvas.tem_sobreposicao(self.canvas, x, y, largura, altura, tipo)
            elif tipo == "diamante":
                sobreposicao = AuxCanvas.tem_sobreposicao(self.canvas, x, y, largura, altura, tipo)
            elif tipo == "circular":
                sobreposicao = AuxCanvas.tem_sobreposicao(self.canvas, x, y, 0, 0, tipo, raio)
            
            # Desenhar apenas se não houver sobreposição
            if not sobreposicao:
                if tipo == "retangular":
                    AuxCanvas.desenhar_retangulo(self.canvas, x, y, largura, altura)
                    print(f"Elemento {indice}: Retângulo adicionado em ({x}, {y})")
                elif tipo == "diamante":
                    AuxCanvas.desenhar_diamante(self.canvas, x, y, largura, altura)
                    print(f"Elemento {indice}: Diamante adicionado em ({x}, {y})")
                elif tipo == "circular":
                    AuxCanvas.desenhar_circulo(self.canvas, x, y, raio)
                    print(f"Elemento {indice}: Círculo adicionado em ({x}, {y})")
            else:
                print(f"Elemento {indice}: Não foi possível adicionar {tipo} - Sobreposição detectada")