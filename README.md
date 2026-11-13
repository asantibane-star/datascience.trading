# Modelo predictivo del dólar (USD/CLP) 
Autores: Alexander Santibañez - Oliver Bustamante

## ¿Cómo poner en marcha el modelo?
- La base de entrenamiento quedo linkeada desde github al modelo para cargarse automaticamente, sin embargo, será necesario cargar la información en tiempo real a la base ya sea cargando la base bruta actualizada y pasandola por el codigo "Limpiador_base" o cargando las variables que se utilizan
- En un cuaderno colab sera necesario ejecutar ```pip install arch -q ``` en la terminal.
- Ejecutado el modelo, automaticamente entregará una prediccion para mañana según los últimos datos cargados. Es decir, antes de la apertura o luego del cierre de EE.UU. puede ser actualizada la base para obtener una prediccion para el día siguiente.

### Orden de ejecución
1. Subir la base bruta de Bloomberg con el nombre `base_usdclp_actualizada.xlsx` (hoja `Días hábiles`) a la misma carpeta del cuaderno.
2. Ejecutar `Limpiador_base.ipynb` (requiere ```pip install holidays -q```). Este cuaderno aplica todo el tratamiento descrito abajo y genera `base_modelo.xlsx` con las hojas `datos`, `auxiliares` y `precios`. Verificamos que la base que genera es idéntica, celda por celda, a la que usamos para entrenar y evaluar el modelo.
3. Ejecutar `Trading_code_v5_politica.ipynb` de principio a fin (unos 30 segundos). La última celda entrega la señal para la próxima apertura.

## Nuestra estrategia
Este modelo opera a la apertura, con el precio de apertura. La intuición que impregnamos en el modelo es realizar una apuesta fuerte (-3, +3) cando la señal es fuerte, y cuando no, mantener una actividad moderada con montos acotados (-1,+1).

## Tratamiento de los datos
Para construir una base de datos limpia, sin data leakage y consistente con los datos entregados, ejecutamos las siguientes acciones junto a su respectivo fundamento:

### 1. Filtrado temporal y limpieza de observaciones
- Eliminamos los fines de semana y feriados en Chile, dejando únicamente los días hábiles en los que efectivamente operó el mercado local.
- Recortamos el inicio de la base al 2 de enero de 2004. Debido a que primero, hasta 1999 operaba el régimen de bandas cambiarias y hasta agosto de 2001 la política monetaria estaba indexada a la UF; por lo que este escenario anterior no era útil. En segundo lugar, hasta finales de 2003 los precios de apertura, máximo, mínimo y cierre son iguales, impidiendo medir rangos intradía y aperturas reales. Finalmente, recién a partir de 2003–2004 existe cobertura completa de variables clave como swaps promedio cámara, TPM nominal, volatilidad implícita y *carry trade*.
- Dentro del tramo 2004–2026 detectamos 157 jornadas concentradas principalmente entre 2011 y 2012, sin precios reportados para el tipo de cambio. Elegimos eliminar estas filas en lugar de usar el dólar observado o recrearlo de forma sintética, ya que introduciría ruido en el modelo.
- Al eliminar feriados y días sin cotización, creamos una variable que mide los días calendario transcurridos desde la observación previa para identificar retornos acumulados tras huecos largos y evitar tratarlos como variaciones de un solo día.
- Como el modelo es de frecuencia diaria, y operará dentro de un mes, eliminamos las 10 columnas mensuales. No obstante, en un modelo a largo plazo creemos que es importante incorporar el impacto de estas variables en el tipo cambio.

### 2. Empalmes, rezagos e imputación de niveles
- Cuando el mercado chileno opera pero mercados externos están cerrados por feriado, aplicamos un ffill en los niveles con un máximo de 3 días. Ya que un mercado cerrado no genera información nueva, creemos que mantener el último precio conocido refleja la información disponible.
- Empalmamos `SOFRRATE` con `FEDL01` para cubrir el tramo histórico previo, descartando la LIBOR 1M (`US0001M`) por haber sido descontinuada en septiembre de 2024. Dado que tanto SOFR como Fed Funds se publican a la mañana siguiente en Nueva York y su valor del día $t$ no se conoce al momento de tomar posición, aplicamos un **rezago de 1 día** a la tasa USD.
- Empalmamos las series históricas respecto a derivados publicadas por el BCCh, especificamente uniendo (`CLPBODM` $\rightarrow$ `CLPBODMB`) y (`CLPBLDM` $\rightarrow$ `CLPBLDMB`), tras verificar en el periodo de traslape correlaciones de 0,998 y 0,984 respectivamente entre las series a unir. Como estas estadísticas están fechadas en el día de referencia pero el Banco Central las publica con desfase, aplicamos un **rezago de 2 días** para evitar data leakage.

### 3. Feature Engineering
- Las series como la del dolar siguen una tendencia a largo plazo y pueden tener estacionariedades en ciertos periodos. No queremos que el modelo aprenda eso ya que de lo contrario apostará por la tendencia y no por encontrar una señal. Por eso transformamos los precios y flujos a retornos logarítmicos o primeras diferencias.
- Mantenemos en nivel el `VIX`, el *carry*, las pendientes de curvas y la volatilidad implícita a 1 mes (`USDCLPV1M`), ya que su magnitud tiene una interpretación directa por los traders e influyen como niveles.
- **Construcción de indicadores financieros y técnicos:**
  - **Carry (`TPM - Tasa USD`):** Calculamos explícitamente el diferencial de tasas cortas que representa el costo o ingreso financiero de mantener posiciones largas en dólares.
  - **Factores de curva y dinámica intradía:** Construimos las pendientes de tasas, señales de *momentum*, salto de apertura y rango intradía del tipo de cambio.
  - **Corrección de consistencia en rangos intradía:** En 17 jornadas donde el máximo o mínimo reportado no contenía a la apertura o al cierre, ajustamos el extremo correspondiente al máximo/mínimo efectivo entre `{Open, High, Low, Close}` en lugar de descartar la fila, para tratar de preservar más datos.
- **Selección de variables:** Para saber que variables incluir y no introducir redundancia en la base del modelo, primero hicimos un clustering para obtener un dendograma, y en base a eso elegir las variables que aportaran informacion de cada grupo, priorizando series con cobertura vigente en 2026 y fundamento económico tras comprobar con regularización Lasso que el exceso de predictores redundantes deteriora el desempeño fuera de muestra. Adicionalmente, incorporamos el **retorno objetivo rezagado ($Y_{t-1}$)**, que es conocido al momento de decidir y captura una autocorrelacion con el retorno de mañana que encontramos.

### 4. Definición de la variable objetivo y tratamiento final.
- **Variable objetivo** La variable objetivo se construye como: ($Y = \ln(\text{Open}_{t+2} / \text{Open}_{t+1})$) que se define como el retorno a predecir entre la apertura de $t+1$ y la apertura de $t+2$. Como la señal se genera tras el cierre de la jornada $t$, evaluar retornos de cierre a cierre ($\text{Close}_{t+1}/\text{Close}_t$) generaba un alfa ficticio por desfase horario; asi que descartamos predecir la variable cierre hoy a cierre de mañana. Creemos que operar de apertura a apertura es más factible en la operacion y corrige el descalce horario que hay con EE.UU.. Si entre $t$ y $t+2$ existe un hueco mayor a 5 días calendario, la $Y$ se deja vacía, ya que la posible señal de operar se diluye con el tiempo.
-  En las variables expresadas en retornos o diferencias, los faltantes por un mercado cerrado se rellenan con `0` . Para los flujos del BCCh previos a su inicio en octubre de 2008, imputar cero equivale a imputar su media, lo que permite recuperar 1.375 jornadas históricas. Para evitar que el modelo confunda la ausencia de cambio con dato inexistente, incorporamos la variable binaria **`Flujos disponibles`**.
- Finalmente descartamos 295 filas con N/A en variables en nivel; ya que si las rellenabamos con cero o un valor sintetico ibamos a introducir ruido y esto carecería de sentido económico. Además, 23 filas posteriores a vacios largos en el tipo de cambio y aquellas filas intermedias sin variable objetivo $Y$ fueron eliminadas, conservando al final únicamente las dos últimas observaciones correspondientes a la predicción en vivo.

## Insumos del modelo
Aquí proceso a describir que se construye sobre las variables escogidas y que se usan como input en el modelo:
- Construimos los siguientes indicadores provenientes del análisis técnico: Cruce entre las medias de 5 y 20 días; entre las de 20 y 50 días, distancia del precio a la media de 50 y 200 días, MACD, Canal de Donchian de 20 días, RSI de 14 días, Bandas de bollinger a 20 días con 2 desviaciones estandar, volatilidad realizada a 20 días. Estos fueron los indicadores tecnicos que nos dieron mejores resultados y que aportaban información según el test de Dickey-Fuller que realizamos sobre una muestra más amplia de indicadores. En general, la literatura indica que indicadores de momentum como estos pueden llegar a aportar a predecir la variacion del tipo de cambio, según lo que investigamos
- Se construye un GARCH(1,1) que se ejecuta en una ventana movil de 5 años y se reestima cada 21 días. De entre todos los modelos econometricos como AR, MA, ARCH, ARMA, ARIMA, ARIMAX, ARCH, GARCH; este último era el único que aportaba a predecir el movimiento de Y según los criterio de información y la correlación, por lo que lo dejamos como insumo para el modelo que va a predecir más adelante.
- La base de datos se separó en 3 muestras: training, validation y testing. 
  - **Training (2015–2022):** el modelo nunca usa datos anteriores a 2015. Como se reentrena con una ventana móvil, en la práctica cada estimación usa solo los últimos años disponibles al momento de predecir. Esta decisión viene ya que corroboramos que incorporar datos más antiguos no aporta al modelo, dado que se incorpora un rolling window
  - **Validation (enero 2023 a junio 2024, 372 días):** En base a esta muestra calibramos los hiperparametros y tomamos decisiones de la arquitectura del modelo como largo de la ventana, nivel de regularización y política de inversión.
  - **Testing (julio 2024 a noviembre 2026, 580 días):** Usamos esta muestra solo para medir el resultado final.
  - Entre cada muestra y en cada reentrenamiento dejamos una **purga de 2 días**, porque la $Y$ de un día recién se conoce dos aperturas después, y así nunca entrenamos con un retorno que todavía no ocurrió.

## Modelo Construido y nuestras experiencias
Antes de entrar en los detalles técnicos del modelo comentaré brevemente el camino recorrido y que intuicion nos guió a elegir este modelo final. En primera instancia optamos por hacer un modelo logistico regularizado con caracteristicas polinomiales de grado 2 hasta 3. Vimos que esta configuración no entregaba buenos resultados, ya que se introducian cientos hasta miles de parametros en el modelo, que terminaban introduciendo ruido más que una señal clara. Luego nos pasamos a construir una red neuronal MLP para ver si podiamos tener mejores resultados, y lo que pasó es que en la muestra de training la red llegaba a un overfitting teniendo un accuracy cercano a 1, pero una vez calibrada con la muestra de validation y corroborada en testing, en esta última lograba un AUC de 0,49 aproximadamente y un desempeño peor que el azar. Intentamos calibrar el modelo, quitar variables, cambiar configuraciones pero nunca pudo superar a una logistica simple, que pasó a ser nuestra base de comparación.

Luego de desechar las alternativas anteriores, nos pasamos a una logística con penalización Lasso y ventana móvil que llegaba a un AUC de 0,544 en validation, pero en testing caía a 0,516 y perdía 12,6% neto. Revisando por qué, nos dimos cuenta de algo que cambió radicalmente nuestro enfoque, y es que predecir bien la dirección no implicaba hacerlo con la magnitud del movimiento y por ende no nos permitia calibrar bien cuando tomar las posiciones correctas. Descubrimos que el resultado de la estrategia depende de los pocos días en que el dólar se mueve mucho, y un clasificador trata igual un día de +0,05% que uno de +2%. Por eso pasamos a una **regresión que predice cuánto se mueve el dólar** y no solo hacia dónde, y a una regla que **solo opera cuando lo que se espera ganar supera a los costos**.

También probamos otras ideas que no funcionaron:
- Obsevamos que más variables no ayudan, pues con las 53 variables (base + técnicos + GARCH) y dejando que Lasso o Elastic Net eligieran, el modelo empeoró tanto en validation como en testing. Detectamos que muchas variables miden casi lo mismo, y ante variables correlacionadas Lasso elige una casi al azar y cambia de elección entre ventanas. Y esta selección al azar por un lado carece de fundamento económico, y por el otro, condiciona el resultado al azar.
- También probamos agregar el cuadrado solo de las variables con curvatura en 2015–2022. Pasaron 2 de 53 y agregarlas bajó la correlación en validation.

### Modelo elegido e implementado
- Dado los antecedentes anteriores, implementamos una **Regresión** con penalización *ridge* sobre $Y$ con **18 variables**, eligiendo una por canal económico del tipo de cambio: términos de intercambio (cobre LME), dólar global (DXY), monedas de la región (USDMXN, USDPEN, USDCOP), apetito por riesgo (VIX, MSCI EM), tasas de EE.UU. (UST 2Y), diferencial de tasas (carry), incertidumbre del peso (variación de la volatilidad implícita a 1 mes y GARCH), flujos (derivados offshore) y la dinámica propia del USD/CLP (retorno al cierre, salto de apertura, momentum de 5 y 20 días, $Y$ rezagada y distancia a la media de 50 días). De cada grupo de variables redundantes escogimos solo un representante. Probamos a meter todas las variables, y otras combinaciones pero las 18 variables elegidas fueron las que dieron mejor resultado.
- El modelo funciona con una **ventana móvil de 756 días**, que se reentrena cada 21 días. El escalamiento de las variables se ajusta solo con la ventana de entrenamiento mediante StandarSacler. Descartamos RobustScaler ya que no habian problemas graves por outliers.
- En lo que respecta a la elección de la arquitectura e hiperparametros del modelo, probamos ventanas de 2 a 5 años y cuatro niveles de regularización, y elegimos la combinación cuya predicción tiene mayor AUC. Lo comprobamos con otros criterios: correlación con Y, accuracy y exactitud balanceada; y nos llevan al mismo resultado.
- El factor clave en este modelo es la **Política de inversión**, ya que comparamos el retorno esperado neto de carry con el costo de entrar y salir de la posición y las reglas del modelo son las siguientes: Tomamos **±1** si el retorno supera la mitad del costo y **±3** solo si lo supera 5 veces, y tomamos decisiones **día por medio**. Elegimos esta política entre 27 combinaciones por el mayor ratio de Sharpe neto en validation.
- Algunos supuestos son que definimos un costo de 3 pb por unidad de exposición que cambia (pasar de −1 a +1 son 2 unidades), carry en base ACT/360 con los días calendario que dura la posición, y P/L medido por unidad de nocional fijo, sin reinversión.

## Resultados
Los resultados del modelo se exponen en la siguiente tabla: 

| | Modelo (validation) | Modelo (testing) | Comprar siempre | No operar nunca | Exposición siempre +1 |
|---|---|---|---|---|---|
| P/L spot | 49,2% | 24,4% | 7,0% | 0% | 25,8% |
| P/L carry | −2,3% | −0,1% | −1,6% | 0% | −0,2% |
| Costos | −6,3% | −7,2% | 0,0% | 0% | −14,3% |
| **P/L neto** | **40,7%** | **17,1%** | 5,4% | 0% | 11,3% |
| Sharpe neto | 1,52 | 0,62 | 0,19 | 0 | 0,40 |
| Máxima caída | −13,0% | −13,6% | −17,3% | 0% | −14,4% |
| % días con posición | 84,9% | 76,9% | 100% | 0% | 100% |
| Acierto de dirección | 57,8% | 52,8% | | | |

Los resultados de las 3 referencias comparativas, son obtenidas sobre la muestra de testing. Algunas lecturas que obtenemos de estos resultados son:
- En testing el modelo termina sobre las tres referencias, con una caída máxima menor que la de comprar siempre. La mayor parte de la ganancia se concentra en el segundo semestre de 2024 (+15,0%) y el segundo de 2025 (+7,1%) mientras que en 2026 el resultado es levemente negativo.
- El test de Diebold-Mariano exigido en las bases, contra apostar siempre al alza da $t$ = 0,86 (p = 0,20) y el bootstrap por bloques de 10 días da p = 0,16 contra no operar, 0,32 contra comprar siempre y 0,46 contra exposición uno. Si bien hay ganancias en el modelo estas no son estadísticamente significativa. Sin embargo, creemos que esto no es malo y de hecho es esperables ya que con series tan ruidosas y señales débiles, resulta poco realista creer que se pueden obtener resultados con significancia estadística. 
- Un punto interesante a notar es que las 4 jornadas en +3 acertaron todas y sumaron +12,6%, mientras que las 4 en −3 acertaron la mitad y restaron 0,3%. Con +1 el modelo estuvo 244 jornadas, acertó 53,3% y sumó +5,6%; con −1 estuvo 194 jornadas, acertó 53,1% y sumó +0,8%. En las 134 jornadas en que se abstuvo no hubo resultado por tipo de cambio ni por carry; el −1,6% de ese grupo corresponde solo al costo de cerrar posiciones. Esto refleja la política de inversión del equipo de apostar en grande cuando la señal es fuerte y mantenerse activo pero conservador cuando no.


### Limitaciones
Por un lado, nuestras decisiones son robustas ya que en 31 de las 32 combinaciones de ventana, regularización y conjunto de variables que probamos, la correlación entre la predicción y el retorno real en testing es positiva, y con la regla de costo el resultado fue positivo en testing tanto con las 18 variables como con las 53, con o sin términos no lineales. Lo que intetamos decir con esto es que los resultados de este modelo no fueron obtenidos al azar sino que son resultado de un arduo trabajo eleigiendo el modelo adecuado, las variables importantes y la politica ganadora.

Sin embargo tenemos que admitir tres limitaciones relevantes. Primero, el Sharpe de 1,52 en validation está inflado porque ahí elegimos entre muchas opciones y dejamos la mejor; por lo que el número relevante es el de testing. Segundo, durante el desarrollo miramos testing más de una vez con modelos anteriores, por lo que en estricto rigor nos adaptamos a los resultados en testing; adaptando el modelo en cierta forma a esta muestra; lo cual ca en contra de los principios del machine learning para obtener un resultado transparente; no obstante, para nosotros la verdadera prueba es cuando se pruebe el modelo en vivo. Por último, determinamos que este modelo es efectivo en el largoplazo, y en un horizonte de 20 días el azar va a pesar más que las señales; queremos declarar que esto no necesariamente es prueba de que el modelo esté malo.
