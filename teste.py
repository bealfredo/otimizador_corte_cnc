import torch
import matplotlib.pyplot as plt

# Dados iniciais (exemplo)
recortes_disponiveis = [
      {"tipo": "retangular", "largura": 29, "altura": 29, "rotacao": 0},
      {"tipo": "retangular", "largura": 29, "altura": 29, "rotacao": 0},
      {"tipo": "retangular", "largura": 29, "altura": 29, "rotacao": 0},
      {"tipo": "retangular", "largura": 29, "altura": 29, "rotacao": 0},
      {"tipo": "retangular", "largura": 139, "altura": 29, "rotacao": 0},
      {"tipo": "retangular", "largura": 60, "altura": 8, "rotacao": 0},
      {"tipo": "retangular", "largura": 44, "altura": 4, "rotacao": 0},
      {"tipo": "diamante", "largura": 29, "altura": 48, "rotacao": 0},
      {"tipo": "diamante", "largura": 29, "altura": 48, "rotacao": 0},
      {"tipo": "diamante", "largura": 29, "altura": 48, "rotacao": 0},
      {"tipo": "circular", "r": 16},
      {"tipo": "circular", "r": 16}
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
        # Verificar se o círculo está completamente dentro do canvas
        if x - raio < 0 or y - raio < 0 or x + raio >= canvas.shape[1] or y + raio >= canvas.shape[0]:
            return True
        
        for i in range(max(0, y - raio), min(canvas.shape[0], y + raio + 1)):
            for j in range(max(0, x - raio), min(canvas.shape[1], x + raio + 1)):
                if (j - x)**2 + (i - y)**2 <= raio**2:
                    if canvas[i, j] == 1:
                        return True
        return False

# Desenhar peças em posições aleatórias
for indice, recorte in enumerate(recortes_disponiveis):
    tipo = recorte["tipo"]
    
    # Definir dimensões baseadas no tipo da peça
    if tipo == "retangular":
        largura = recorte["largura"]
        altura = recorte["altura"]
        x_max = largura_canvas - largura
        y_max = altura_canvas - altura
    elif tipo == "diamante":
        largura = recorte["largura"]
        altura = recorte["altura"]
        x_max = largura_canvas - largura
        y_max = altura_canvas - altura
    elif tipo == "circular":
        raio = recorte["r"]
        x_max = largura_canvas - 2 * raio
        y_max = altura_canvas - 2 * raio
    
    # Garantir valores não negativos
    x_max = max(1, x_max)
    y_max = max(1, y_max)
    
    # Gerar posição aleatória
    x = torch.randint(0, x_max, (1,), device='cuda').item()
    y = torch.randint(0, y_max, (1,), device='cuda').item()
    
    # Verificar sobreposição
    sobreposicao = False
    if tipo == "retangular":
        sobreposicao = tem_sobreposicao(canvas, x, y, largura, altura, tipo)
    elif tipo == "diamante":
        sobreposicao = tem_sobreposicao(canvas, x, y, largura, altura, tipo)
    elif tipo == "circular":
        sobreposicao = tem_sobreposicao(canvas, x, y, 0, 0, tipo, raio)
    
    # Desenhar apenas se não houver sobreposição
    if not sobreposicao:
        if tipo == "retangular":
            desenhar_retangulo(canvas, x, y, largura, altura)
            print(f"Elemento {indice}: Retângulo adicionado em ({x}, {y})")
        elif tipo == "diamante":
            desenhar_diamante(canvas, x, y, largura, altura)
            print(f"Elemento {indice}: Diamante adicionado em ({x}, {y})")
        elif tipo == "circular":
            desenhar_circulo(canvas, x, y, raio)
            print(f"Elemento {indice}: Círculo adicionado em ({x}, {y})")
    else:
        print(f"Elemento {indice}: Não foi possível adicionar {tipo} - Sobreposição detectada")

# Visualizar o resultado
canvas_np = canvas.cpu().numpy()
plt.figure(figsize=(10, 5))
plt.imshow(canvas_np, cmap='gray')
plt.title("Canvas com Peças em Posições Aleatórias")
plt.axis('on')
plt.grid(True, alpha=0.3)
plt.show()