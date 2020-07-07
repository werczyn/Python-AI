import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#droga hamowania w zaleznosci od wilgotnosci i predkosci

#Przygotowanie wartosci

x_wilgotnosc = np.arange(0, 11, 1)
x_predkosc = np.arange(0, 141, 10)
x_droga = np.arange(0, 101, 1)

wilgotnosc_lo = fuzz.trimf(x_wilgotnosc, [0, 0, 5])
wilgotnosc_md = fuzz.trimf(x_wilgotnosc, [0, 5, 10])
wilgotnosc_hi = fuzz.trimf(x_wilgotnosc, [5, 10, 10])
predkosc_lo = fuzz.trimf(x_predkosc, [0, 0, 40])
predkosc_md = fuzz.trimf(x_predkosc, [0, 40, 80])
predkosc_hi = fuzz.trimf(x_predkosc, [80, 140, 140])
droga_lo = fuzz.trimf(x_droga, [0, 0, 30])
droga_md = fuzz.trimf(x_droga, [0, 30, 60])
droga_hi = fuzz.trimf(x_droga, [60, 100, 100])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_wilgotnosc, wilgotnosc_lo, 'b', linewidth=1.5, label='Niska')
ax0.plot(x_wilgotnosc, wilgotnosc_md, 'g', linewidth=1.5, label='Średnia')
ax0.plot(x_wilgotnosc, wilgotnosc_hi, 'r', linewidth=1.5, label='Wysoka')
ax0.set_title('Wilgotnosc')
ax0.legend()

ax1.plot(x_predkosc, predkosc_lo, 'b', linewidth=1.5, label='Wolno')
ax1.plot(x_predkosc, predkosc_md, 'g', linewidth=1.5, label='Akceptowalnie')
ax1.plot(x_predkosc, predkosc_hi, 'r', linewidth=1.5, label='Szybko')
ax1.set_title('Predkosc')
ax1.legend()

ax2.plot(x_droga, droga_lo, 'b', linewidth=1.5, label='Krótka')
ax2.plot(x_droga, droga_md, 'g', linewidth=1.5, label='Normalna')
ax2.plot(x_droga, droga_hi, 'r', linewidth=1.5, label='Długa')
ax2.set_title('Droga hamowania')
ax2.legend()

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------
#Wprowadzanie reguł

wilgotnosc_level_lo = fuzz.interp_membership(x_wilgotnosc, wilgotnosc_lo, 1.5)
wilgotnosc_level_md = fuzz.interp_membership(x_wilgotnosc, wilgotnosc_md, 1.5)
wilgotnosc_level_hi = fuzz.interp_membership(x_wilgotnosc, wilgotnosc_hi, 1.5)

predkosc_level_lo = fuzz.interp_membership(x_predkosc, predkosc_lo, 20.8)
predkosc_level_md = fuzz.interp_membership(x_predkosc, predkosc_md, 20.8)
predkosc_level_hi = fuzz.interp_membership(x_predkosc, predkosc_hi, 20.8)


droga_activation_lo = np.fmin(predkosc_level_lo, droga_lo)

active_rule2 = np.fmax(predkosc_level_md, predkosc_level_md)
droga_activation_md = np.fmin(active_rule2, droga_md)

active_rule3 = np.fmax(wilgotnosc_level_hi, predkosc_level_hi)
droga_activation_hi = np.fmin(active_rule3, droga_hi)

active_rule4 = np.fmax(wilgotnosc_level_md, predkosc_level_hi)
droga_activation_4 = np.fmin(active_rule4, droga_hi)


droga0 = np.zeros_like(x_droga)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_droga, droga0, droga_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_droga, droga_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_droga, droga0, droga_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_droga, droga_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_droga, droga0, droga_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_droga, droga_hi, 'r', linewidth=0.5, linestyle='--')

ax0.fill_between(x_droga, droga0, droga_activation_4, facecolor='r', alpha=0.7)
ax0.plot(x_droga, droga_hi, 'm', linewidth=0.5, linestyle='--')

ax0.set_title('Wyjściowa przynależność:')

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------
#Agregacja, Defuzzifuzifikacja

# Aggregate all three output membership functions together
aggregated = np.fmax(droga_activation_4,
             np.fmax(droga_activation_lo,
                     np.fmax(droga_activation_md, droga_activation_hi)))

# Calculate defuzzified result
droga = fuzz.defuzz(x_droga, aggregated, 'centroid')
droga_activation = fuzz.interp_membership(x_droga, aggregated, droga)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_droga, droga_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_droga, droga_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_droga, droga_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_droga, droga0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([droga, droga], [0, droga_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

plt.tight_layout()
plt.show()
