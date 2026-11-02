"""Los cuatro modelos, de unos 100k parámetros cada uno. Todos reciben (N, 1, 28, 28)
y cada uno construye su propia vista: vector, imagen, secuencia de filas o secuencia de parches."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Sin sesgo estructural: el orden de los 784 píxeles le es indiferente."""
    def __init__(self, hidden=112):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    @property
    def first_layer(self):
        return self.net[1]

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Localidad y pesos compartidos; el global average pooling la acerca a la invariancia a traslaciones."""
    def __init__(self, channels=(32, 64, 136)):
        super().__init__()
        c1, c2, c3 = channels
        block = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.features = nn.Sequential(*block(1, c1), nn.MaxPool2d(2),
                                      *block(c1, c2), nn.MaxPool2d(2),
                                      *block(c2, c3))
        self.head = nn.Linear(c3, 10)

    @property
    def first_layer(self):
        return self.features[0]

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


class RowLSTM(nn.Module):
    """Orden secuencial: lee la imagen fila a fila (28 pasos de 28 píxeles) con los mismos pesos."""
    def __init__(self, hidden=144):
        super().__init__()
        self.lstm = nn.LSTM(28, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 10)

    def forward(self, x):
        _, (h, _) = self.lstm(x.reshape(x.shape[0], 28, 28))
        return self.head(h[-1])


def patchify(x, p=4):
    """(N,1,28,28) -> (N, 49, 16): parches p×p en orden de lectura."""
    n, g = x.shape[0], 28 // p
    return x.reshape(n, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(n, g * g, p * p)


class ViT(nn.Module):
    """Atención entre todos los parches. Sin posiciones sería equivariante a permutar tokens;
    con posiciones aprendidas, la posición de cada parche se aprende de los datos."""
    def __init__(self, dim=64, depth=3, heads=4, ff=128, patch=4, use_pos=True):
        super().__init__()
        self.patch, self.use_pos = patch, use_pos
        tokens = (28 // patch) ** 2
        self.embed = nn.Linear(patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, tokens + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def tokens(self, x):
        t = self.embed(patchify(x, self.patch))
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        return t + self.pos if self.use_pos else t

    def forward(self, x):
        return self.head(self.norm(self.encoder(self.tokens(x)))[:, 0])


BUILDERS = {"mlp": MLP, "cnn": CNN, "lstm": RowLSTM, "vit": ViT}


def build(name):
    return BUILDERS[name]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
