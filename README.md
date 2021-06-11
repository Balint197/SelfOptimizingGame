# A06_SelfOptimizingGame

A programot a [Godot Engine](https://godotengine.org/download) segítségével készítettük - a futtatáshoz csak ennek a letöltésére van szükség (standard version).

Egyéb eszközök: 
* [Godot-Python](https://github.com/touilleMan/godot-python) - python kódot tudunk használni a Godot-ban
* [numpy](https://numpy.org/)

## A program felépítése:

* `track3.tscn` - a fő jelenet (pálya), ezen belül helyezkedik el a többi node
* `wall.tscn` - fal, amivel ütköznek az autók
* `finish.tscn` - célvonal
* `path2d` - a pálya útját követi, ez alapján értékeljük ki a megtett út távolságát
* `camera2d` - megjelenítésért és a kisebb felbontásért felelős
* `controller.tscn` - node, ami vezérli az elemeket, ebben van a legfontosabb logika (PSO is)
* `car_AI` - a controller által létrehozott tanuló autó példánya

A többi elem egyéb próbálkozások, kísérletezések eredményei, vagy ezen elemek által használt node-ok. 

A `controller.py` paraméterei alapján létrejönnek az autók, és a PSO algoritmus alapján megindul a tanulás. 

## A tanítás paraméterei

Az autók kimenete a sebesség és az 5 db raycast távolsága logaritmikusan leskálázva:

[sebesség log(raycast(1:5))]

Az autók bemenete:

[gyorsulás kanyarodás]

, ahol a gyorsulás [-1...1], és a [-1...-0.3] értékek teljes fékezést, a [-0.3...0.3] gurulást (enyhe lassulással), és [0.3...1] teljes gyorsulást jelent - "digitális" gázadás modell. A kanyarodás analóg módon, [-1...1]*maxkanyarodás elven működik.

A fitnesz függvény a megtett távolság, és az eltelt idő alapján adódik. 

A következő populáció akkor következik, ha az autók már nem mozognak (megálltak, vagy falba ütköztek). A teljes szimuláció véget ér egy maximum iteráció elérésével, vagy ha az összes autó célba ért. 

A PSO algoritmus egyedei egy-egy autóhoz vannak rendezve, és be- illetve kimenetei a korábban említett paraméterek. 

## A PSO működése

Ebben a projektben az autókat a bemenetek alapján irányító neurális hálók súlyait a Particle Swarm Optimization vagy részecske-raj optimalizáció (PSO) algoritmussal optimalizáltuk. Míg a genetikus algoritmusok, evolúciós stratégiák és más módszerek a biológiai evolúciót veszik mintának, a PSO egyedek szociális interakcióját szimulálja, mint például repülő rovar- vagy madárrajok, innen származik az elnevezés is. Minden egyed, PSO-terminológiában részecske, figyelembe veszi a saját és a raj többit tagjának mozgását az optimalizáció során.

A feladat során egy módosított PSO algoritmust alkalmaztunk, amelyet a következő egyenletrendszer ír le:
```math
\begin{align}
v_i^{k+1}&=w\cdot v_i^k+c_1\cdot \textrm{rand}() \cdot (\textrm{pbest}_i^k-x_i^k)+c_2\cdot \textrm{rand}() \cdot (\textrm{gbest}^k-x_i^k)\\
x_i^{k+1}&=x_i^k+v_i^{k+1}
\end{align}
```
Itt $`v`$ jelöli a részecske sebességét, $`x`$ a helyzetét, $`i`$ a részecske sorszáma, $`k`$ az epochok száma. Az adott részecske által elért legjobb megoldást `pbest`, a teljes raj által elért legjobb megoldást `gbest` jelöli. Ez a módosított algoritmus tartalmaz egy $`w`$ súlyt, amely egyfajta tehetetlenségként szerepel, $`w`$>1 esetén a leglassabb globális optimum elérése, de lassabban szűkül be a keresési terület, 1-nél kisebb érték esetén gyorsan beszűkül a keresési terület, de kevesebb iteráció alatt éri el a globális optimumot, amennyiben az a keresési területen található. 

Jelen esetben a részecskék az egyes autókat irányító neurális hálók. A $`c`$<sub>2</sub> és $`w`$ súlyok nem konstansak, ha egy autó célba ér, a $`w`$ tehetetlenség 1-ről 0,994-re csökken, a $`c`$<sub>2</sub> (globális optmimumhoz tartozó) súly pedig 8-szorosára növekszik. A $`v`$ vektor, mivel minden tanulási lépésben $`w`$ tehetetlenséggel van megszorozva, ezért 1-nél kisebb tehetetlenség és megfelelő iterációszám esetén a $`v`$ 0-ra fog csökkenni, és az adott részecske csakis a legjobbak irányába mozog tovább. Ugyanezen okból nem szabad $`w`$>1 értéket választani, ugyanis akkor a sebességvektor a végtelenbe fog tartani.

## Eredmények
![](/data/evdata.png)


