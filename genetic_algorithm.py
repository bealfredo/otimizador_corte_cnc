from common.layout_display import LayoutDisplayMixin
import torch
import matplotlib.pyplot as plt
import copy
import random
import math
import time

# Its done

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=150, taxa_mutacao=0.2):
        print("Algoritmo Genético para Otimização do Corte de Chapa. Executado por Marco.")
        self.TAM_POP = TAM_POP
        self.initial_layout = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []
        self.POP_AUX = []
        self.aptidao = []
        self.geracoes_totais = numero_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.melhor_aptidoes = []
        self.optimized_layout = []
        self.pecas_nao_colocadas = []
        self.elite_size = max(1, int(TAM_POP * 0.1))  # 10% de elitismo

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilizando dispositivo: {self.device}")
        
        self.matriz_chapa = torch.zeros((sheet_height, sheet_width), device=self.device, dtype=torch.bool)
        self.area_total_chapa = sheet_width * sheet_height
        
        # Pré-calcula áreas das peças
        self.areas_pecas = {}
        for recorte in recortes_disponiveis:
            tipo = recorte["tipo"]
            key = (tipo, recorte.get("largura", 0), recorte.get("altura", 0), recorte.get("r", 0))
            if tipo == "retangular":
                self.areas_pecas[key] = recorte["largura"] * recorte["altura"]
            elif tipo == "diamante":
                self.areas_pecas[key] = (recorte["largura"] * recorte["altura"]) / 2
            elif tipo == "circular":
                self.areas_pecas[key] = math.pi * recorte["r"] ** 2

    def initialize_population(self, recortes_disponiveis):
        """Inicializa população com todos os recortes disponíveis em posições aleatórias."""
        self.POP = []
        for _ in range(self.TAM_POP):
            individuo = []
            
            for recorte in recortes_disponiveis:
                peca_nova = copy.deepcopy(recorte)
                
                # Calcular posições aleatórias válidas
                tipo = peca_nova["tipo"]
                if tipo == "retangular":
                    largura, altura = peca_nova["largura"], peca_nova["altura"]
                    x_max = self.sheet_width - largura
                    y_max = self.sheet_height - altura
                elif tipo == "diamante":
                    largura, altura = peca_nova["largura"], peca_nova["altura"]
                    x_max = self.sheet_width - largura
                    y_max = self.sheet_height - altura
                elif tipo == "circular":
                    raio = peca_nova["r"]
                    x_max = self.sheet_width - 2 * raio
                    y_max = self.sheet_height - 2 * raio
                
                x_max = max(1, x_max)
                y_max = max(1, y_max)
                peca_nova["x"] = torch.randint(0, x_max, (1,), device=self.device).item()
                peca_nova["y"] = torch.randint(0, y_max, (1,), device=self.device).item()
                
                individuo.append(peca_nova)
                
            self.POP.append(individuo)
        
        self.aptidao = torch.zeros(self.TAM_POP, device=self.device)

    def evaluate_batch(self):
        """Avalia a população em lote com otimizações para melhor desempenho."""
        import concurrent.futures
        from multiprocessing import cpu_count
        
        # Inicializa arrays para armazenar resultados
        batch_size = self.TAM_POP
        areas_ocupadas = torch.zeros(batch_size, device=self.device)
        sobreposicoes = torch.zeros(batch_size, device=self.device)
        fora_limites = torch.zeros(batch_size, device=self.device)
        compactness = torch.zeros(batch_size, device=self.device)  # Mede quão compacto está o layout
        corner_proximity = torch.zeros(batch_size, device=self.device)  # NOVO: Mede proximidade ao canto
        
        # Função para avaliar um único indivíduo
        def avaliar_individuo(idx):
            # Criar matriz local para evitar condições de corrida
            matriz = torch.zeros((self.sheet_height, self.sheet_width), device="cpu", dtype=torch.bool)
            individuo = self.POP[idx]
            area_ocupada = 0
            sobreposicao = 0
            fora_limite = 0
            
            # Obter centróide e dimensões para cálculos
            pontos_x = []
            pontos_y = []
            
            aux_canvas = AuxCanvas(self.sheet_height, self.sheet_width, self.sheet_width, 
                                  self.sheet_height, matriz, "cpu")
            
            for peca in individuo:
                tipo = peca["tipo"]
                x, y = peca["x"], peca["y"]
                
                # Atualizar pontos para cálculo de compactness
                if tipo == "retangular":
                    largura, altura = peca["largura"], peca["altura"]
                    centro_x = x + largura/2
                    centro_y = y + altura/2
                    pontos_x.append(centro_x)
                    pontos_y.append(centro_y)
                    
                    if x < 0 or y < 0 or x + largura > self.sheet_width or y + altura > self.sheet_height:
                        fora_limite += 1
                        continue
                    
                    if not aux_canvas.tem_sobreposicao(matriz, x, y, largura, altura, tipo):
                        aux_canvas.desenhar_retangulo(matriz, x, y, largura, altura, peca)
                        area_ocupada += self.areas_pecas[(tipo, largura, altura, 0)]
                    else:
                        sobreposicao += 1
                        
                elif tipo == "diamante":
                    largura, altura = peca["largura"], peca["altura"]
                    centro_x = x + largura/2
                    centro_y = y + altura/2
                    pontos_x.append(centro_x)
                    pontos_y.append(centro_y)
                    
                    if x < 0 or y < 0 or x + largura > self.sheet_width or y + altura > self.sheet_height:
                        fora_limite += 1
                        continue
                    
                    if not aux_canvas.tem_sobreposicao(matriz, x, y, largura, altura, tipo):
                        aux_canvas.desenhar_diamante(matriz, x, y, largura, altura, peca)
                        area_ocupada += self.areas_pecas[(tipo, largura, altura, 0)]
                    else:
                        sobreposicao += 1
                        
                elif tipo == "circular":
                    raio = peca["r"]
                    centro_x = x + raio
                    centro_y = y + raio
                    pontos_x.append(centro_x)
                    pontos_y.append(centro_y)
                    
                    if x < 0 or y < 0 or x + 2*raio > self.sheet_width or y + 2*raio > self.sheet_height:
                        fora_limite += 1
                        continue
                    
                    if not aux_canvas.tem_sobreposicao(matriz, x, y, 0, 0, tipo, raio):
                        aux_canvas.desenhar_circulo(matriz, x, y, raio, peca)
                        area_ocupada += self.areas_pecas[(tipo, 0, 0, raio)]
                    else:
                        sobreposicao += 1
            
            # Calcular compactness se houver pelo menos 2 peças
            compact = 0
            if len(pontos_x) >= 2:
                # Calcular distância média entre todas as peças
                total_dist = 0
                count = 0
                for i in range(len(pontos_x)):
                    for j in range(i+1, len(pontos_x)):
                        dist = ((pontos_x[i] - pontos_x[j])**2 + (pontos_y[i] - pontos_y[j])**2)**0.5
                        total_dist += dist
                        count += 1
                
                if count > 0:
                    avg_dist = total_dist / count
                    # Normalização: quanto menor a distância média, maior o valor de compactness
                    compact = 1 / (1 + avg_dist / max(self.sheet_width, self.sheet_height))
            
            # Calcular proximidade apenas ao canto inferior esquerdo
            corner_prox = 0
            if pontos_x and pontos_y:
                # Calcular o centróide médio de todas as peças
                avg_x = sum(pontos_x) / len(pontos_x)
                avg_y = sum(pontos_y) / len(pontos_y)
                
                # Canto específico (inferior esquerdo)
                corner_x, corner_y = 0, 0
                
                # Calcular distância a este canto específico
                dist = ((avg_x - corner_x)**2 + (avg_y - corner_y)**2)**0.5
                
                # Normalizar: quanto menor a distância ao canto, maior o valor
                diagonal = ((self.sheet_width)**2 + (self.sheet_height)**2)**0.5
                corner_prox = 1 - (dist / diagonal)  # Valor entre 0 e 1
                    
            return idx, area_ocupada, sobreposicao, fora_limite, compact, corner_prox
        
        # Usar multithreading para avaliar indivíduos em paralelo
        max_workers = min(cpu_count() + 1, batch_size)
        resultados = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(avaliar_individuo, i) for i in range(batch_size)]
            for future in concurrent.futures.as_completed(futures):
                idx, area, sobrep, fora, comp, corner = future.result()
                areas_ocupadas[idx] = area
                sobreposicoes[idx] = sobrep
                fora_limites[idx] = fora
                compactness[idx] = comp
                corner_proximity[idx] = corner
                resultados.append((idx, area, sobrep, fora, comp, corner))
        
        # Calcular aptidão final com ponderações ajustadas
        validas = torch.tensor([len(ind) for ind in self.POP], device=self.device) - sobreposicoes - fora_limites
        
        # Pesos
        w1 = 5.0   # Peso para peças válidas (maior prioridade)
        w2 = 2.0   # Peso para área ocupada (importante)
        w3 = 2.0   # Peso para penalização de sobreposições e peças fora dos limites
        w4 = 1.5   # Peso para compactness (moderado)
        w5 = 2.0   # NOVO: Peso para proximidade ao canto (importante)
        
        # Calcular aptidão final, agora incluindo proximidade ao canto
        self.aptidao = (w1 * validas) + (w2 * areas_ocupadas) - (w3 * (sobreposicoes + fora_limites)) + \
                      (w4 * compactness) + (w5 * corner_proximity)

    def selection(self):
        pais = []
        indices_ordenados = torch.argsort(self.aptidao, descending=True)
        for i in range(min(self.elite_size, self.TAM_POP)):
            pais.append(self.POP[indices_ordenados[i].item()])
        
        while len(pais) < self.TAM_POP // 2:
            torneio_indices = random.sample(range(self.TAM_POP), min(3, self.TAM_POP))
            melhor_idx = max(torneio_indices, key=lambda i: self.aptidao[i].item())
            pais.append(self.POP[melhor_idx])
        return pais

    def crossover(self, pai1, pai2):
        tamanho_min = min(len(pai1), len(pai2))
        filho = []
        
        # Para cada peça, escolher aleatoriamente entre pai1 and pai2
        for i in range(tamanho_min):
            peca_escolhida = random.choice([pai1[i], pai2[i]])
            filho.append(copy.deepcopy(peca_escolhida))
            
        return filho

    def mutation(self, individuo):
        """Mutação aleatória nas posições de algumas peças."""
        for i in range(len(individuo)):
            if random.random() < self.taxa_mutacao:
                peca = individuo[i]
                tipo = peca["tipo"]
                
                # Calcular limites para posicionamento válido
                if tipo == "retangular":
                    largura, altura = peca["largura"], peca["altura"]
                    x_max = self.sheet_width - largura
                    y_max = self.sheet_height - altura
                elif tipo == "diamante":
                    largura, altura = peca["largura"], peca["altura"]
                    x_max = self.sheet_width - largura
                    y_max = self.sheet_height - altura
                elif tipo == "circular":
                    raio = peca["r"]
                    x_max = self.sheet_width - 2 * raio
                    y_max = self.sheet_height - 2 * raio
                
                # Garantir que x_max and y_max são pelo menos 1
                x_max = max(1, x_max)
                y_max = max(1, y_max)
                
                # Gerar novas coordenadas aleatórias
                peca["x"] = torch.randint(0, x_max, (1,), device=self.device).item()
                peca["y"] = torch.randint(0, y_max, (1,), device=self.device).item()
                
        return individuo

    def genetic_operators(self):
        pais = self.selection()
        self.POP_AUX = [copy.deepcopy(self.POP[i]) for i in torch.argsort(self.aptidao, descending=True)[:self.elite_size].tolist()]
        
        while len(self.POP_AUX) < self.TAM_POP:
            pai1, pai2 = random.sample(pais, 2)
            filho = self.crossover(pai1, pai2)
            filho = self.mutation(filho)
            self.POP_AUX.append(filho)
        self.POP = self.POP_AUX

    def run(self):
        """Executa o algoritmo genético com otimizações de performance."""
        import concurrent.futures
        from multiprocessing import cpu_count
        
        print("\n=== Iniciando Algoritmo Genético com processamento paralelo ===")
        recortes_disponiveis = copy.deepcopy(self.initial_layout)
        
        # Número máximo de workers para processamento paralelo
        max_workers = min(cpu_count() + 1, 16)  # Limitar para não sobrecarregar o sistema
        print(f"Usando até {max_workers} threads para processamento paralelo")
        
        # Inicializar população com todos os recortes disponíveis
        print(f"Inicializando população com {len(recortes_disponiveis)} peças...")
        self.initialize_population(recortes_disponiveis)
        
        # Evoluir a população
        start_time = time.time()
        best_fitness_history = []
        best_individual = None
        best_fitness = float('-inf')
        
        print(f"Iniciando evolução por {self.geracoes_totais} gerações...")
        for geracao in range(self.geracoes_totais):
            # Avaliar população em paralelo
            self.evaluate_batch()
            
            # Registrar melhor aptidão
            melhor_apt = torch.max(self.aptidao).item()
            best_fitness_history.append(melhor_apt)
            
            # Verificar se encontramos um novo melhor indivíduo
            current_best_idx = torch.argmax(self.aptidao).item()
            if melhor_apt > best_fitness:
                best_fitness = melhor_apt
                best_individual = copy.deepcopy(self.POP[current_best_idx])
                
            # Exibir progresso
            if geracao % 10 == 0 or geracao == self.geracoes_totais - 1:
                elapsed = time.time() - start_time
                print(f"Geração {geracao+1}/{self.geracoes_totais} - Melhor aptidão: {melhor_apt:.2f} - Tempo: {elapsed:.2f}s")
            
            # Aplicar operadores genéticos se não for a última geração
            if geracao < self.geracoes_totais - 1:
                self.genetic_operators()
        
        print(f"\nEvolução concluída em {time.time() - start_time:.2f} segundos")
        print(f"Melhor aptidão encontrada: {best_fitness:.2f}")
        
        # Encontrar peças válidas no melhor indivíduo
        valid_layout = self.encontrar_todas_pecas_validas(best_individual)
        num_pecas_validas = len(valid_layout)
        
        print(f"Peças válidas colocadas: {num_pecas_validas}/{len(recortes_disponiveis)}")
        
        # Identificar peças que não foram colocadas
        self.pecas_nao_colocadas = []
        pecas_colocadas_tipos = [(p["tipo"], p.get("largura", 0), p.get("altura", 0), p.get("r", 0)) for p in valid_layout]
        
        for recorte in recortes_disponiveis:
            tipo = recorte["tipo"]
            if tipo == "retangular" or tipo == "diamante":
                identificador = (tipo, recorte["largura"], recorte["altura"], 0)
            else:  # circular
                identificador = (tipo, 0, 0, recorte["r"])
                
            # Se encontramos uma peça do mesmo tipo nas peças válidas, removemos da lista
            if identificador in pecas_colocadas_tipos:
                idx = pecas_colocadas_tipos.index(identificador)
                pecas_colocadas_tipos.pop(idx)
            else:
                self.pecas_nao_colocadas.append(copy.deepcopy(recorte))
        
        # Desenhar o melhor layout encontrado
        self.matriz_chapa.zero_()
        aux_canvas = AuxCanvas(self.sheet_height, self.sheet_width, self.sheet_width, 
                              self.sheet_height, self.matriz_chapa, self.device)
        
        for peca in valid_layout:
            tipo = peca["tipo"]
            if tipo == "retangular":
                aux_canvas.desenhar_retangulo(self.matriz_chapa, peca["x"], peca["y"], 
                                           peca["largura"], peca["altura"], peca)
            elif tipo == "diamante":
                aux_canvas.desenhar_diamante(self.matriz_chapa, peca["x"], peca["y"], 
                                          peca["largura"], peca["altura"], peca)
            elif tipo == "circular":
                aux_canvas.desenhar_circulo(self.matriz_chapa, peca["x"], peca["y"], 
                                           peca["r"], peca)
        
        # Mostrar relatório final
        print(f"\n=== Relatório Final ===")
        print(f"Peças posicionadas: {num_pecas_validas}/{len(recortes_disponiveis)}")
        print(f"Peças não posicionadas: {len(self.pecas_nao_colocadas)}")
        if self.pecas_nao_colocadas:
            print("Detalhes das peças não posicionadas:")
            for i, peca in enumerate(self.pecas_nao_colocadas):
                tipo = peca["tipo"]
                if tipo == "retangular" or tipo == "diamante":
                    print(f"  {i+1}. {tipo} {peca['largura']}x{peca['altura']}")
                elif tipo == "circular":
                    print(f"  {i+1}. {tipo} raio={peca['r']}")
        
        # Exibir evolução da aptidão
        plt.figure(figsize=(10, 4))
        plt.plot(best_fitness_history)
        plt.title("Evolução da Aptidão")
        plt.xlabel("Geração")
        plt.ylabel("Melhor Aptidão")
        plt.grid(True)
        plt.show()
        
        # Exibir layout final
        self.optimized_layout = valid_layout
        self.display_layout(self.optimized_layout, title="Melhor Layout Encontrado")
        
        return self.optimized_layout

    def optimize_and_display(self):
        """Método principal para executar o algoritmo and exibir resultados."""
        print("Starting optimization...")
        self.display_layout(self.initial_layout, title="Initial Layout")
        self.optimized_layout = self.run()
        # self.display_layout(self.optimized_layout, title="Optimized Layout")
        return self.optimized_layout

    def encontrar_todas_pecas_validas(self, individuo):
        """Encontra todas as peças válidas em um único indivíduo, sem sobreposições."""
        matriz_temp = torch.zeros((self.sheet_height, self.sheet_width), device=self.device, dtype=torch.bool)
        aux_canvas = AuxCanvas(self.sheet_height, self.sheet_width, self.sheet_width, 
                              self.sheet_height, matriz_temp, self.device)
        
        pecas_validas = []
        
        # Ordenar peças considerando área e distância aos cantos
        pecas_ordenadas = []
        corners = [
            (0, 0),                           # Canto inferior esquerdo
            (0, self.sheet_height),           # Canto superior esquerdo
            (self.sheet_width, 0),            # Canto inferior direito
            (self.sheet_width, self.sheet_height)  # Canto superior direito
        ]
        
        for peca in individuo:
            tipo = peca["tipo"]
            x, y = peca["x"], peca["y"]
            
            # Calcular centro da peça
            if tipo == "retangular" or tipo == "diamante":
                largura, altura = peca["largura"], peca["altura"]
                centro_x = x + largura/2
                centro_y = y + altura/2
                if tipo == "retangular":
                    area = peca["largura"] * peca["altura"]
                else:
                    area = (peca["largura"] * peca["altura"]) / 2
            elif tipo == "circular":
                raio = peca["r"]
                centro_x = x + raio
                centro_y = y + raio
                area = 3.1416 * peca["r"] ** 2
            else:
                centro_x, centro_y = x, y
                area = 0
            
            # Encontrar o canto mais próximo
            min_corner_dist = float('inf')
            for corner_x, corner_y in corners:
                dist = ((centro_x - corner_x)**2 + (centro_y - corner_y)**2)**0.5
                min_corner_dist = min(min_corner_dist, dist)
            
            # Calcular pontuação composta: área - distância ao canto (normalizada)
            # Isso favorece peças grandes e próximas aos cantos
            diagonal = ((self.sheet_width)**2 + (self.sheet_height)**2)**0.5
            corner_factor = min_corner_dist / diagonal  # Normalizado entre 0 e 1
            pontuacao = area - (corner_factor * area * 0.5)  # Redução de até 50% baseada na distância
            
            pecas_ordenadas.append((peca, pontuacao))
        
        # Ordenar por pontuação decrescente
        pecas_ordenadas.sort(key=lambda x: x[1], reverse=True)
        
        # Processar peças na ordem de pontuação
        for peca_tuple in pecas_ordenadas:
            peca = peca_tuple[0]
            tipo = peca["tipo"]
            x, y = peca["x"], peca["y"]
            
            # Verificar validade
            valida = True
            
            if tipo == "retangular":
                largura, altura = peca["largura"], peca["altura"]
                if x < 0 or y < 0 or x + largura > self.sheet_width or y + altura > self.sheet_height:
                    valida = False
                else:
                    sobreposicao = aux_canvas.tem_sobreposicao(matriz_temp, x, y, largura, altura, tipo)
                    if not sobreposicao:
                        aux_canvas.desenhar_retangulo(matriz_temp, x, y, largura, altura, peca)
                        pecas_validas.append(copy.deepcopy(peca))
                    
            elif tipo == "diamante":
                largura, altura = peca["largura"], peca["altura"]
                if x < 0 or y < 0 or x + largura > self.sheet_width or y + altura > self.sheet_height:
                    valida = False
                else:
                    sobreposicao = aux_canvas.tem_sobreposicao(matriz_temp, x, y, largura, altura, tipo)
                    if not sobreposicao:
                        aux_canvas.desenhar_diamante(matriz_temp, x, y, largura, altura, peca)
                        pecas_validas.append(copy.deepcopy(peca))
                    
            elif tipo == "circular":
                raio = peca["r"]
                if x < 0 or y < 0 or x + 2*raio > self.sheet_width or y + 2*raio > self.sheet_height:
                    valida = False
                else:
                    sobreposicao = aux_canvas.tem_sobreposicao(matriz_temp, x, y, 0, 0, tipo, raio)
                    if not sobreposicao:
                        aux_canvas.desenhar_circulo(matriz_temp, x, y, raio, peca)
                        pecas_validas.append(copy.deepcopy(peca))
        
        return pecas_validas


class AuxCanvas:
    def __init__(self, altura_canvas, largura_canvas, sheet_width, sheet_height, canvas, device):
        self.altura_canvas = altura_canvas
        self.largura_canvas = largura_canvas
        self.canvas = canvas
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.device = device

    @staticmethod
    def desenhar_retangulo(canvas, x, y, largura, altura, recorte):
        x, y = int(x), int(y)
        largura, altura = int(largura), int(altura)
        y_start = canvas.shape[0] - y - altura
        y_end = canvas.shape[0] - y
        x_start, x_end = x, min(x + largura, canvas.shape[1])
        if x_start >= 0 and y_start >= 0 and x_end <= canvas.shape[1] and y_end <= canvas.shape[0]:
            canvas[y_start:y_end, x_start:x_end] = 1

    @staticmethod
    def desenhar_diamante(canvas, x, y, largura, altura, recorte):
        x, y = int(x), int(y)
        largura, altura = int(largura), int(altura)
        metade_largura = largura // 2
        metade_altura = altura // 2
        centro_x = x + metade_largura
        centro_y = canvas.shape[0] - y - metade_altura
        for j in range(max(0, x), min(canvas.shape[1], x + largura)):
            for i in range(max(0, canvas.shape[0] - y - altura), min(canvas.shape[0], canvas.shape[0] - y)):
                if (abs(j - centro_x) / metade_largura + abs(i - centro_y) / metade_altura) <= 1:
                    canvas[i, j] = 1

    @staticmethod
    def desenhar_circulo(canvas, x, y, raio, recorte):
        x, y = int(x), int(y)
        raio = int(raio)
        centro_x = x + raio
        centro_y = canvas.shape[0] - y - raio
        for j in range(max(0, centro_x - raio), min(canvas.shape[1], centro_x + raio + 1)):
            for i in range(max(0, centro_y - raio), min(canvas.shape[0], centro_y + raio + 1)):
                if (j - centro_x)**2 + (i - centro_y)**2 <= raio**2:
                    canvas[i, j] = 1

    @staticmethod
    def tem_sobreposicao(canvas, x, y, largura, altura, tipo, raio=None):
        """
        Verifica se há sobreposição entre uma peça and o canvas.
        Implementa verificação preliminar usando bounding boxes para maior eficiência.
        """
        # Primeiro definimos o retângulo delimitador (bounding box) para cada tipo
        if tipo == "retangular":
            bb_x_start, bb_y_start = x, y
            bb_x_end, bb_y_end = x + largura, y + altura
        elif tipo == "diamante":
            bb_x_start, bb_y_start = x, y
            bb_x_end, bb_y_end = x + largura, y + altura
        elif tipo == "circular":
            bb_x_start, bb_y_start = x, y
            bb_x_end, bb_y_end = x + 2*raio, y + 2*raio
        
        # Convertendo para coordenadas da matriz (origem no canto superior esquerdo)
        matrix_bb_y_start = canvas.shape[0] - bb_y_end  # Note a inversão devido à diferença de sistema de coordenadas
        matrix_bb_y_end = canvas.shape[0] - bb_y_start
        matrix_bb_x_start, matrix_bb_x_end = bb_x_start, bb_x_end
        
        # Verificar limites da chapa
        if (matrix_bb_x_start < 0 or matrix_bb_y_start < 0 or 
            matrix_bb_x_end > canvas.shape[1] or matrix_bb_y_end > canvas.shape[0]):
            return True  # Fora dos limites da chapa
        
        # Verificação rápida: se não houver nenhum pixel ocupado no retângulo delimitador,
        # não há necessidade de verificação detalhada
        if not torch.any(canvas[matrix_bb_y_start:matrix_bb_y_end, matrix_bb_x_start:matrix_bb_x_end]):
            return False  # Não há sobreposição
        
        # Se chegamos aqui, há pixels ocupados no bounding box, precisamos verificar em detalhes
        if tipo == "retangular":
            # Para retângulos, o bounding box é a própria forma, então já sabemos que há sobreposição
            return True
        elif tipo == "diamante":
            metade_largura = largura // 2
            metade_altura = altura // 2
            centro_x = x + metade_largura
            centro_y = y + metade_altura
            centro_matrix_y = canvas.shape[0] - centro_y  # Ajuste para coordenadas da matriz
            
            # Verificação pixel a pixel dentro do bounding box
            for j in range(matrix_bb_x_start, matrix_bb_x_end):
                for i in range(matrix_bb_y_start, matrix_bb_y_end):
                    # Verificamos se o pixel está dentro do diamante
                    # Convertendo de volta para o sistema de coordenadas original para cálculo
                    pixel_x = j
                    pixel_y = canvas.shape[0] - i
                    # Equação do diamante: |x-x0|/a + |y-y0|/b <= 1
                    if ((abs(pixel_x - centro_x) / metade_largura + 
                         abs(pixel_y - centro_y) / metade_altura) <= 1):
                        if canvas[i, j]:
                            return True
            return False
        elif tipo == "circular":
            centro_x = x + raio
            centro_y = y + raio
            centro_matrix_y = canvas.shape[0] - centro_y  # Ajuste para coordenadas da matriz
            
            # Verificação pixel a pixel dentro do bounding box
            for j in range(matrix_bb_x_start, matrix_bb_x_end):
                for i in range(matrix_bb_y_start, matrix_bb_y_end):
                    # Verificamos se o pixel está dentro do círculo
                    # Convertendo de volta para o sistema de coordenadas original para cálculo
                    pixel_x = j
                    pixel_y = canvas.shape[0] - i
                    # Equação do círculo: (x-x0)² + (y-y0)² <= r²
                    if ((pixel_x - centro_x)**2 + (pixel_y - centro_y)**2 <= raio**2):
                        if canvas[i, j]:
                            return True
            return False
