from common.layout_display import LayoutDisplayMixin
import torch
import matplotlib.pyplot as plt
import copy


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
        self.melhor_aptidoes = []
        self.optimized_layout = None  # To be set after optimization

        # Configura dispositivo (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matriz_chapa = torch.zeros((sheet_height, sheet_width), device=self.device, dtype=torch.bool)
        
        # Initialize population after setting up the device
        self.initialize_population()

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
        try:
            print("Running genetic algorithm visualization...")
            auxCanvas = AuxCanvas(self.sheet_height, self.sheet_width, self.sheet_width, self.sheet_height, self.matriz_chapa, self.device)
            placed_pieces = auxCanvas.desenhar_peca_aleatoria(self.initial_layout)

            print(f"Successfully placed {len(placed_pieces)} pieces")
            
            # Visualizar o resultado - 0,0 no canto inferior esquerdo
            canvas_np = self.matriz_chapa.cpu().numpy() if self.device.type == 'cuda' else self.matriz_chapa.numpy()
            plt.figure(figsize=(10, 5))
            plt.imshow(canvas_np, cmap='gray')
            plt.title("Canvas com Peças em Posições Aleatórias")
            plt.axis('on')
            plt.grid(True, alpha=0.3)
            plt.show()
            # Set the optimized layout to the placed pieces
            self.optimized_layout = placed_pieces
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            # Ensure we return something even if there's an error
            self.optimized_layout = self.initial_layout
            
        return self.optimized_layout

    def optimize_and_display(self):
        """
        Displays the initial layout, runs the optimization algorithm,
        and displays the optimized layout using the mixin's display_layout method.
        """
        # Display initial layout
        print("Starting optimization and display process...")
        self.display_layout(self.initial_layout, title="Initial Layout - Genetic Algorithm")
        
        # Run the optimization algorithm
        self.optimized_layout = self.run()
        
        # Display optimized layout
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm")
        return self.optimized_layout
    

class AuxCanvas():
    def __init__(self, altura_canvas, largura_canvas, sheet_width, sheet_height, canvas, device=None):
        self.altura_canvas = altura_canvas
        self.largura_canvas = largura_canvas
        self.canvas = canvas
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Funções para desenhar formas
    @staticmethod
    def desenhar_retangulo(canvas, x, y, largura, altura, recorte):
        """
        Draw a rectangle on the canvas.
        Adjusted to use bottom-left origin system (like LayoutDisplayMixin).
        """
        x, y = int(x), int(y)
        largura, altura = int(largura), int(altura)
        
        # Calculate the top boundary in matrix coordinates (y increases downward in matrix)
        y_matrix_start = canvas.shape[0] - y - altura  # Convert from bottom-left to top-left origin
        y_matrix_end = canvas.shape[0] - y  # Convert from bottom-left to top-left origin
        
        x_start = x
        x_end = min(x + largura, canvas.shape[1])
        
        # Check bounds
        if x_start < 0 or y_matrix_start < 0 or x_end > canvas.shape[1] or y_matrix_end > canvas.shape[0]:
            # Out of bounds, don't draw
            pass
        else:
            # Draw the rectangle - note reversed y coordinates
            canvas[y_matrix_start:y_matrix_end, x_start:x_end] = 1
            
        # Return the updated piece with its coordinates in the LayoutDisplayMixin system
        resultado = copy.deepcopy(recorte)
        resultado["x"] = x
        resultado["y"] = y  # This is already in bottom-left coordinate system
        # Make sure rotation is preserved
        if "rotacao" in recorte:
            resultado["rotacao"] = recorte["rotacao"]
        return resultado

    @staticmethod
    def desenhar_diamante(canvas, x, y, largura, altura, recorte):
        """
        Draw a diamond on the canvas.
        Adjusted to use bottom-left origin system (like LayoutDisplayMixin).
        """
        x, y = int(x), int(y)
        largura, altura = int(largura), int(altura)
        metade_largura = largura // 2
        metade_altura = altura // 2
        
        # Center point in matrix coordinates
        centro_x = x + metade_largura
        centro_y = canvas.shape[0] - y - metade_altura  # Adjust for bottom-left origin
        
        # Draw the diamond
        for j in range(max(0, x), min(canvas.shape[1], x + largura)):
            for i in range(max(0, canvas.shape[0] - y - altura), min(canvas.shape[0], canvas.shape[0] - y)):
                # Convert to matrix coordinates for calculation
                matrix_y = i
                matrix_x = j
                # Check if point is within diamond
                if (abs(matrix_x - centro_x) / metade_largura + abs(matrix_y - centro_y) / metade_altura) <= 1:
                    canvas[matrix_y, matrix_x] = 1
                    
        # Return the updated piece with its coordinates in the LayoutDisplayMixin system
        resultado = copy.deepcopy(recorte)
        resultado["x"] = x
        resultado["y"] = y  # This is already in bottom-left coordinate system
        # Make sure rotation is preserved
        if "rotacao" in recorte:
            resultado["rotacao"] = recorte["rotacao"]
        return resultado

    @staticmethod
    def desenhar_circulo(canvas, x, y, raio, recorte):
        """
        Draw a circle on the canvas.
        Adjusted to use bottom-left origin system (like LayoutDisplayMixin).
        """
        x, y = int(x), int(y)
        raio = int(raio)
        
        # Center point in matrix coordinates
        centro_x = x + raio
        centro_y = canvas.shape[0] - y - raio  # Adjust for bottom-left origin
        
        # Draw the circle
        for j in range(max(0, centro_x - raio), min(canvas.shape[1], centro_x + raio + 1)):
            for i in range(max(0, centro_y - raio), min(canvas.shape[0], centro_y + raio + 1)):
                # Check if point is within circle
                if (j - centro_x)**2 + (i - centro_y)**2 <= raio**2:
                    canvas[i, j] = 1
                    
        # Return the updated piece with its coordinates in the LayoutDisplayMixin system
        resultado = copy.deepcopy(recorte)
        resultado["x"] = x
        resultado["y"] = y  # This is already in bottom-left coordinate system
        return resultado

    # Função para verificar sobreposição
    @staticmethod
    def tem_sobreposicao(canvas, x, y, largura, altura, tipo, raio=None):
        """
        Check for overlap with existing pieces on the canvas.
        Adjusted to use bottom-left origin system.
        """
        if tipo == "retangular":
            # Convert coordinates to matrix system (top-left origin)
            y_matrix_start = canvas.shape[0] - y - altura
            y_matrix_end = canvas.shape[0] - y
            x_start = x
            x_end = min(x + largura, canvas.shape[1])
            
            # Check bounds
            if x_start < 0 or y_matrix_start < 0 or x_end > canvas.shape[1] or y_matrix_end > canvas.shape[0]:
                return True  # Out of bounds
                
            # Check for overlap
            region = canvas[y_matrix_start:y_matrix_end, x_start:x_end]
            return torch.any(region == 1)
        
        elif tipo == "diamante":
            metade_largura = largura // 2
            metade_altura = altura // 2
            centro_x = x + metade_largura
            centro_y = canvas.shape[0] - y - metade_altura  # Adjust for bottom-left origin
            
            # Check for overlap
            for j in range(max(0, x), min(canvas.shape[1], x + largura)):
                for i in range(max(0, canvas.shape[0] - y - altura), min(canvas.shape[0], canvas.shape[0] - y)):
                    matrix_y = i
                    matrix_x = j
                    if (abs(matrix_x - centro_x) / metade_largura + abs(matrix_y - centro_y) / metade_altura) <= 1:
                        if canvas[matrix_y, matrix_x] == 1:
                            return True
            return False
        
        elif tipo == "circular":
            # Convert center to matrix coordinates
            centro_x = x + raio
            centro_y = canvas.shape[0] - y - raio  # Adjust for bottom-left origin
            
            # Check if circle is within canvas bounds
            if centro_x - raio < 0 or centro_y - raio < 0 or centro_x + raio >= canvas.shape[1] or centro_y + raio >= canvas.shape[0]:
                return True
            
            # Check for overlap
            for j in range(max(0, centro_x - raio), min(canvas.shape[1], centro_x + raio + 1)):
                for i in range(max(0, centro_y - raio), min(canvas.shape[0], centro_y + raio + 1)):
                    if (j - centro_x)**2 + (i - centro_y)**2 <= raio**2:
                        if canvas[i, j] == 1:
                            return True
            return False
        
    def desenhar_peca_aleatoria(self, recortes_disponiveis):
        recortes_inseridos = []
        
        # Desenhar peças em posições aleatórias
        for indice, recorte in enumerate(recortes_disponiveis):
            try:
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
                
                # Gerar posição aleatória usando o device correto
                x = torch.randint(0, x_max, (1,), device=self.device).item()
                y = torch.randint(0, y_max, (1,), device=self.device).item()
                
                # Verificar sobreposição
                sobreposicao = False
                if tipo == "retangular":
                    sobreposicao = self.tem_sobreposicao(self.canvas, x, y, largura, altura, tipo)
                elif tipo == "diamante":
                    sobreposicao = self.tem_sobreposicao(self.canvas, x, y, largura, altura, tipo)
                elif tipo == "circular":
                    sobreposicao = self.tem_sobreposicao(self.canvas, x, y, 0, 0, tipo, raio)
                
                # Desenhar apenas se não houver sobreposição
                if not sobreposicao:
                    if tipo == "retangular":
                        updated_piece = self.desenhar_retangulo(self.canvas, x, y, largura, altura, recorte)
                        novo_recorte = recorte.copy()
                        novo_recorte["x"] = x
                        novo_recorte["y"] = y
                        recortes_inseridos.append(novo_recorte)
                        print(f"Elemento {indice}: Retângulo adicionado em ({x}, {y}), tamanho: {largura}x{altura}")
                    elif tipo == "diamante":
                        updated_piece = self.desenhar_diamante(self.canvas, x, y, largura, altura, recorte)
                        novo_recorte = recorte.copy()
                        novo_recorte["x"] = x
                        novo_recorte["y"] = y
                        recortes_inseridos.append(novo_recorte)
                        print(f"Elemento {indice}: Diamante adicionado em ({x}, {y}), tamanho: {largura}x{altura}")
                    elif tipo == "circular":
                        updated_piece = self.desenhar_circulo(self.canvas, x, y, raio, recorte)
                        novo_recorte = recorte.copy()
                        novo_recorte["x"] = x
                        novo_recorte["y"] = y
                        recortes_inseridos.append(novo_recorte)
                        print(f"Elemento {indice}: Círculo adicionado em ({x}, {y}), raio: {raio}")
                else:
                    print(f"Elemento {indice}: Não foi possível adicionar {tipo} - Sobreposição detectada")
            except Exception as e:
                print(f"Error processing element {indice}: {str(e)}")
                
        return recortes_inseridos