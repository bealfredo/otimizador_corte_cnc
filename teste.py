import torch

# Dados iniciais (exemplo)
recortes_disponiveis = [
    {"tipo": "retangular", "largura": 29, "altura": 29, "x": 1, "y": 1, "rotacao": 0},
    {"tipo": "retangular", "largura": 29, "altura": 29, "x": 31, "y": 1, "rotacao": 0},
    {"tipo": "diamante", "largura": 29, "altura": 48, "x": 32, "y": 31, "rotacao": 0},
    {"tipo": "circular", "r": 16, "x": 124, "y": 2}
]

# Configurações do canvas
altura_canvas = 100
largura_canvas = 200
canvas = torch.zeros((altura_canvas, largura_canvas), dtype=torch.uint8, device='cuda')

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
        for i in range(max(0, y - raio), min(canvas.shape[0], y + raio + 1)):
            for j in range(max(0, x - raio), min(canvas.shape[1], x + raio + 1)):
                if (j - x)**2 + (i - y)**2 <= raio**2:
                    if canvas[i, j] == 1:
                        return True
        return False

# Adicionar peças iniciais
for recorte in recortes_disponiveis:
    tipo = recorte["tipo"]
    x = recorte["x"]
    y = recorte["y"]
    if tipo == "retangular":
        desenhar_retangulo(canvas, x, y, recorte["largura"], recorte["altura"])
    elif tipo == "diamante":
        desenhar_diamante(canvas, x, y, recorte["largura"], recorte["altura"])
    elif tipo == "circular":
        desenhar_circulo(canvas, x, y, recorte["r"])

# Adicionar peças em posições aleatórias
num_pecas_aleatorias = 5
for _ in range(num_pecas_aleatorias):
    # Escolher uma peça aleatória da lista
    recorte = recortes_disponiveis[torch.randint(0, len(recortes_disponiveis), (1,)).item()]
    tipo = recorte["tipo"]
    
    # Gerar posição aleatória
    x_max = largura_canvas - (recorte.get("largura", recorte.get("r", 0)) or 0)
    y_max = altura_canvas - (recorte.get("altura", recorte.get("r", 0)) or 0)
    x = torch.randint(0, max(1, x_max), (1,), device='cuda').item()
    y = torch.randint(0, max(1, y_max), (1,), device='cuda').item()

    # Verificar sobreposição
    if tipo == "retangular":
        largura, altura = recorte["largura"], recorte["altura"]
        if not tem_sobreposicao(canvas, x, y, largura, altura, tipo):
            desenhar_retangulo(canvas, x, y, largura, altura)
            print(f"Peça retangular adicionada em ({x}, {y})")
        else:
            print(f"Sobreposição detectada em ({x}, {y}) para retângulo")
    
    elif tipo == "diamante":
        largura, altura = recorte["largura"], recorte["altura"]
        if not tem_sobreposicao(canvas, x, y, largura, altura, tipo):
            desenhar_diamante(canvas, x, y, largura, altura)
            print(f"Peça diamante adicionada em ({x}, {y})")
        else:
            print(f"Sobreposição detectada em ({x}, {y}) para diamante")
    
    elif tipo == "circular":
        raio = recorte["r"]
        if not tem_sobreposicao(canvas, x, y, 2*raio, 2*raio, tipo, raio):
            desenhar_circulo(canvas, x, y, raio)
            print(f"Peça circular adicionada em ({x}, {y})")
        else:
            print(f"Sobreposição detectada em ({x}, {y}) para círculo")

# Visualizar (opcional)
import matplotlib.pyplot as plt
canvas_np = canvas.cpu().numpy()
plt.imshow(canvas_np, cmap='gray')
plt.title("Máscara com Peças Aleatórias")
plt.show()