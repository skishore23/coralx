**Category-Theory cheat-sheet in pictures**

---

## 0 · Legend (applies to all sketches)

```
(A)          round node  = object
A -->|f| B   labelled arrow = morphism
...dotted... extra structure (cone legs, iso)
```

<br>

## 1 · Mono / Epi / Iso

```mermaid
graph LR
  A((A)) -->|"m (mono)"| B((B))
  C((C)) -->|"e (epi)"| D((D))
  E((E)) -->|"i (iso)"| F((F))
  F -->|"i⁻¹"| E
  classDef mono stroke-width:3,stroke-dasharray:0 0;
  classDef epi stroke-width:1,stroke-dasharray:5 5;
  class A,B mono
  class C,D epi
```

*Bold arrow* = **monomorphism** (left-cancellable);
dashed = **epimorphism** (right-cancellable);
double-headed pair = **isomorphism**.

---

## 2 · Product  $A × B$

```mermaid
graph LR
  subgraph C
    P(["A×B"])
    A((A))
    B((B))
    P -->|"π₁"| A
    P -->|"π₂"| B
    X((X))
    X -.->|f| A
    X -.->|g| B
    X -->|"⟨f,g⟩"| P
  end
```

`⟨f,g⟩` is the **unique morphism** making the outer triangles commute – that’s the universal property of a **product**.

---

## 3 · Coproduct  $A + B$

```mermaid
graph LR
  subgraph C
    A((A))
    B((B))
    S(["A+B"])
    A -->|"ι₁"| S
    B -->|"ι₂"| S
    S -.->|h| Y((Y))
    A -->|f| Y
    B -->|g| Y
  end
```

`h` factors **uniquely** through the injections; dual to the product.

---

## 4 · Pullback  (fibre-product)

```mermaid
graph TD
  PB((P))
  A((A)) -->|f| C((C))
  B((B)) -->|g| C
  PB -->|"p₁"| A
  PB -->|"p₂"| B
  classDef pb stroke-dasharray:0 0,stroke-width:3;
  class PB pb
```

`P` plus arrows `p₁, p₂` make the square commute and are **universal**—give *any* other commuting square you get a unique map into `P`.

---

## 5 · Pushout  (dual of pullback)

```mermaid
graph TD
  A((A)) -->|f| C((C))
  A -->|g| B((B))
  C -->|"ι₁"| PO((Q))
  B -->|"ι₂"| PO
```

Replace “fibre” with “co-fibre” and arrows with duals.

---

## 6 · Terminal & Initial Objects

```mermaid
graph LR
  subgraph C
    T((1)):::term
    I((0)):::init
    X((X))
    X -->|"!"| T
    I -->|"!!"| X
  end
  classDef term stroke-width:3;
  classDef init stroke-dasharray:5 5;
```

*Exactly one* arrow **to** the terminal `1`;
*exactly one* arrow **from** the initial `0`.

---

## 7 · Adjunction  $F ⊣ U$

```mermaid
graph LR
  subgraph D
    FY(["F X"])
    Y((Y))
  end
  subgraph C
    X((X))
    UX(["U Y"])
  end
  X -->|F| FY
  UX -->|"F⊣U"| Y
  FY -.->|"η"| UX
  X -.->|"ε"| Y
```

Natural bijection

$$
\text{Hom}_\mathbf{D}(F X,\,Y) \;\cong\; \text{Hom}_\mathbf{C}(X,\,U Y)
$$

`η` = unit, `ε` = counit.

---

## 8 · Free / Forgetful (classic adjunction)

```mermaid
graph LR
  subgraph Set
    S((S))
  end
  subgraph Mon
    FS(["Free(S)"])
  end
  S -->|"F (free)"| FS
  FS -->|"U (forget)"| S
```

`F ⊣ U` : sets ⟷ monoids.
This picture generalises to **free groups**, **free categories**, etc.

---

## 9 · Monad from Adjunction

```mermaid
graph LR
  subgraph C
    X((X))
    TTX(["T² X"])
    TX(["T X"])
  end
  X -->|"η"| TX
  TX -->|"μ"| X
  TX -->|T| TTX
```

`T = U F`, `η` (unit) and `μ = U ε F` (multiplication) satisfy the monad laws.

---

## 10 · Limit Cone (general pattern)

```mermaid
graph TD
  L(["Lim D"])
  subgraph "Diagram D"
    A((A))
    B((B))
    C((C))
    A -->|"d₁"| B
    B -->|"d₂"| C
  end
  L -.->|"λA"| A
  L -.->|"λB"| B
  L -.->|"λC"| C
```

Any other cone factors *uniquely* through `L`.  Replace arrows with duals to picture a colimit.

---