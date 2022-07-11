import matplotlib.pyplot as plt
import numpy as np

import scan.simulations as sims

np.random.seed(100)
# Lorenz:
Lorenz63 = sims.Lorenz63().simulate(2)
Roessler = sims.Roessler().simulate(2)
ComplexButterly = sims.ComplexButterly().simulate(2)
Chen = sims.Chen().simulate(2)
ChuaCircuit = sims.ChuaCircuit().simulate(2)
Thomas = sims.Thomas().simulate(2)
WindmiAttractor = sims.WindmiAttractor().simulate(2)
Rucklidge = sims.Rucklidge().simulate(2)
Henon = sims.Henon().simulate(2)
Logistic = sims.Logistic().simulate(2)
SimplestDrivenChaotic = sims.SimplestDrivenChaotic().simulate(2)
UedaOscillator = sims.UedaOscillator().simulate(2)
KuramotoSivashinsky = sims.KuramotoSivashinsky().simulate(2)
Lorenz96 = sims.Lorenz96().simulate(500)

print(Lorenz63)
print(Roessler)
print(ComplexButterly)
print(Chen)
print(ChuaCircuit)
print(Thomas)
print(WindmiAttractor)
print(Rucklidge)
print(Henon)
print(Logistic)
print(SimplestDrivenChaotic)
print(UedaOscillator)
print(KuramotoSivashinsky)
print(Lorenz96)

# print(Lorenz63)
# print(Lorenz63)
# print(Lorenz63)
# print(Lorenz63)
# print(Lorenz63)
# print(Lorenz63)
# print(Lorenz63)

plt.scatter(Logistic[:-1, 0], Logistic[1:, 0])
plt.show()

plt.scatter(Henon[:, 0], Henon[:, 1])
plt.show()

plt.plot(Lorenz63[:, 0], Lorenz63[:, 1])
plt.show()

plt.plot(Roessler[:, 0], Roessler[:, 1])
plt.show()

plt.plot(ComplexButterly[:, 0], ComplexButterly[:, 1])
plt.show()

plt.plot(Chen[:, 0], Chen[:, 1])
plt.show()

plt.plot(ChuaCircuit[:, 0], ChuaCircuit[:, 1])
plt.show()

plt.plot(Thomas[:, 0], Thomas[:, 1])
plt.show()

plt.plot(WindmiAttractor[:, 0], WindmiAttractor[:, 1])
plt.show()

plt.plot(Rucklidge[:, 0], Rucklidge[:, 1])
plt.show()

plt.plot(SimplestDrivenChaotic[:, 0], SimplestDrivenChaotic[:, 1])
plt.show()

plt.plot(UedaOscillator[:, 0], UedaOscillator[:, 1])
plt.show()

plt.imshow(KuramotoSivashinsky.T, aspect="auto")
plt.show()

plt.imshow(Lorenz96.T, aspect="auto")
plt.show()

plt.plot([0], [0])
plt.show()

np.set_printoptions(precision=15)

# print(Lorenz96[1])

print(np.array2string(Lorenz96[1], precision=17, separator=', '))
# help(sims.Lorenz63())
