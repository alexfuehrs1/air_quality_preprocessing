import pandas as pd
import matplotlib.pyplot as plt
import os

TARGET = "eaqi_partial_class"

# CSV einlesen (Pfad anpassen!)
path = os.path.abspath(os.path.join(os.path.dirname(__file__),))
df = pd.read_csv(path + "/ml_ready_eaqi.csv", parse_dates=["ts"])

# Alle Klassen 1-6 erzwingen
all_classes = range(1, 7)

# Klassen zählen
counts = df[TARGET].value_counts().reindex(all_classes, fill_value=0)
proportions = counts / counts.sum()

# Absolut
plt.figure(figsize=(8,5))
counts.plot(kind="bar")
plt.title("Klassenverteilung EAQI (absolut)")
plt.xlabel("Klasse")
plt.ylabel("Anzahl")
plt.xticks(range(len(all_classes)), all_classes)  # sicherstellen, dass 1–6 auf Achse stehen
plt.savefig(path + "/classes/eaqi_klassen_absolut.png", dpi=300)

# Relativ
plt.figure(figsize=(8,5))
proportions.plot(kind="bar")
plt.title("Klassenverteilung EAQI (relativ)")
plt.xlabel("Klasse")
plt.ylabel("Anteil")
plt.xticks(range(len(all_classes)), all_classes)
plt.savefig(path + "/classes/eaqi_klassen_relative.png", dpi=300)