"""
Bot de Trading Automatizado - Alpaca Paper Trading
Estrategia: EMA 9/21 + RSI
Activos: Acciones y Crypto
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN — edita estos valores
# ─────────────────────────────────────────────

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

# Activos a monitorear
STOCKS = ["AAPL", "TSLA", "NVDA"]       # Acciones
CRYPTO = ["BTC/USD", "ETH/USD"]         # Crypto

# Parámetros de estrategia
EMA_SHORT    = 9       # EMA rápida
EMA_LONG     = 21      # EMA lenta
RSI_PERIOD   = 14      # Periodo RSI
RSI_BUY      = 45      # RSI mínimo para comprar
RSI_SELL     = 65      # RSI máximo (señal de venta)

# Gestión de capital
RISK_PER_TRADE = 0.05  # 5% del portafolio por operación
MAX_POSITIONS  = 5     # Máximo de posiciones abiertas simultáneas

# Intervalo de revisión (segundos)
CHECK_INTERVAL = 60    # Revisa señales cada 60 segundos

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLIENTES ALPACA
# ─────────────────────────────────────────────

trading_client      = TradingClient(API_KEY, SECRET_KEY, paper=True)
stock_data_client   = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client  = CryptoHistoricalDataClient()  # Crypto no requiere claves para datos

# ─────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────────

def calcular_ema(series: pd.Series, periodo: int) -> pd.Series:
    return series.ewm(span=periodo, adjust=False).mean()

def calcular_rsi(series: pd.Series, periodo: int = 14) -> pd.Series:
    delta = series.diff()
    ganancia = delta.clip(lower=0).rolling(periodo).mean()
    perdida  = (-delta.clip(upper=0)).rolling(periodo).mean()
    rs = ganancia / perdida
    return 100 - (100 / (1 + rs))

def obtener_señal(df: pd.DataFrame) -> str:
    """
    Retorna 'BUY', 'SELL' o 'HOLD' basado en EMA + RSI.
    Señal de compra:  EMA9 cruza por encima de EMA21 y RSI < RSI_BUY threshold
    Señal de venta:   EMA9 cruza por debajo de EMA21 o RSI > RSI_SELL threshold
    """
    df = df.copy()
    df["ema_short"] = calcular_ema(df["close"], EMA_SHORT)
    df["ema_long"]  = calcular_ema(df["close"], EMA_LONG)
    df["rsi"]       = calcular_rsi(df["close"], RSI_PERIOD)

    ultima   = df.iloc[-1]
    anterior = df.iloc[-2]

    cruce_alcista = (anterior["ema_short"] <= anterior["ema_long"]) and \
                    (ultima["ema_short"] > ultima["ema_long"])
    cruce_bajista = (anterior["ema_short"] >= anterior["ema_long"]) and \
                    (ultima["ema_short"] < ultima["ema_long"])

    rsi_ok_compra = ultima["rsi"] < RSI_BUY
    rsi_ok_venta  = ultima["rsi"] > RSI_SELL

    if cruce_alcista and rsi_ok_compra:
        return "BUY"
    elif cruce_bajista or rsi_ok_venta:
        return "SELL"
    return "HOLD"

# ─────────────────────────────────────────────
# DATOS DE MERCADO
# ─────────────────────────────────────────────

def obtener_barras_stock(symbol: str, dias: int = 30) -> pd.DataFrame | None:
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            start=datetime.now() - timedelta(days=dias),
        )
        barras = stock_data_client.get_stock_bars(request)
        df = barras.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        return df.reset_index()
    except Exception as e:
        log.error(f"Error obteniendo datos de {symbol}: {e}")
        return None

def obtener_barras_crypto(symbol: str, dias: int = 30) -> pd.DataFrame | None:
    try:
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            start=datetime.now() - timedelta(days=dias),
        )
        barras = crypto_data_client.get_crypto_bars(request)
        df = barras.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        return df.reset_index()
    except Exception as e:
        log.error(f"Error obteniendo datos de {symbol}: {e}")
        return None

# ─────────────────────────────────────────────
# GESTIÓN DE ÓRDENES Y POSICIONES
# ─────────────────────────────────────────────

def obtener_posiciones() -> dict:
    """Retorna un dict {symbol: cantidad} de posiciones abiertas."""
    try:
        posiciones = trading_client.get_all_positions()
        return {p.symbol: float(p.qty) for p in posiciones}
    except Exception as e:
        log.error(f"Error obteniendo posiciones: {e}")
        return {}

def obtener_capital_disponible() -> float:
    try:
        cuenta = trading_client.get_account()
        return float(cuenta.cash)
    except Exception as e:
        log.error(f"Error obteniendo cuenta: {e}")
        return 0.0

def calcular_cantidad(precio: float, capital: float) -> float:
    """Calcula cuántas unidades comprar basado en % de riesgo."""
    monto = capital * RISK_PER_TRADE
    return round(monto / precio, 4)

def ejecutar_orden(symbol: str, side: OrderSide, qty: float):
    try:
        orden = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
        )
        resultado = trading_client.submit_order(orden)
        log.info(f"✅ Orden enviada: {side.value.upper()} {qty} {symbol} | ID: {resultado.id}")
        return resultado
    except Exception as e:
        log.error(f"❌ Error ejecutando orden {symbol}: {e}")
        return None

def precio_actual(symbol: str, es_crypto: bool) -> float | None:
    """Obtiene el último precio de cierre disponible."""
    try:
        if es_crypto:
            df = obtener_barras_crypto(symbol, dias=2)
        else:
            df = obtener_barras_stock(symbol, dias=2)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
    except Exception as e:
        log.error(f"Error obteniendo precio de {symbol}: {e}")
    return None

# ─────────────────────────────────────────────
# LÓGICA PRINCIPAL DEL BOT
# ─────────────────────────────────────────────

def procesar_activo(symbol: str, es_crypto: bool):
    log.info(f"📊 Analizando {symbol}...")

    # Obtener datos históricos
    df = obtener_barras_crypto(symbol, dias=30) if es_crypto \
         else obtener_barras_stock(symbol, dias=30)

    if df is None or len(df) < EMA_LONG + 5:
        log.warning(f"⚠️  Datos insuficientes para {symbol}")
        return

    señal = obtener_señal(df)
    precio = float(df["close"].iloc[-1])
    rsi_val = round(calcular_rsi(df["close"], RSI_PERIOD).iloc[-1], 1)

    log.info(f"   {symbol} | Precio: ${precio:.2f} | RSI: {rsi_val} | Señal: {señal}")

    posiciones = obtener_posiciones()
    tiene_posicion = symbol.replace("/", "") in posiciones or symbol in posiciones

    if señal == "BUY" and not tiene_posicion:
        if len(posiciones) >= MAX_POSITIONS:
            log.info(f"   ⏸️  Máximo de posiciones alcanzado ({MAX_POSITIONS})")
            return
        capital = obtener_capital_disponible()
        qty = calcular_cantidad(precio, capital)
        if qty > 0:
            ejecutar_orden(symbol, OrderSide.BUY, qty)

    elif señal == "SELL" and tiene_posicion:
        sym_key = symbol.replace("/", "") if symbol.replace("/", "") in posiciones else symbol
        qty = posiciones.get(sym_key, 0)
        if qty > 0:
            ejecutar_orden(symbol, OrderSide.SELL, qty)

def mostrar_resumen():
    """Muestra el estado actual del portafolio."""
    try:
        cuenta    = trading_client.get_account()
        posiciones = trading_client.get_all_positions()
        log.info("=" * 50)
        log.info(f"💼 PORTAFOLIO — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        log.info(f"   Capital disponible: ${float(cuenta.cash):,.2f}")
        log.info(f"   Valor total:        ${float(cuenta.portfolio_value):,.2f}")
        log.info(f"   P&L hoy:            ${float(cuenta.equity) - float(cuenta.last_equity):,.2f}")
        if posiciones:
            log.info("   Posiciones abiertas:")
            for p in posiciones:
                log.info(f"     • {p.symbol}: {p.qty} unidades | P&L: ${float(p.unrealized_pl):.2f}")
        else:
            log.info("   Sin posiciones abiertas.")
        log.info("=" * 50)
    except Exception as e:
        log.error(f"Error mostrando resumen: {e}")

# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def main():
    log.info("🚀 Bot de Trading iniciado — Modo PAPER TRADING")
    log.info(f"   Acciones: {STOCKS}")
    log.info(f"   Crypto:   {CRYPTO}")
    log.info(f"   Estrategia: EMA{EMA_SHORT}/{EMA_LONG} + RSI{RSI_PERIOD}")
    log.info(f"   Riesgo por operación: {RISK_PER_TRADE*100:.0f}%")

    ciclo = 0
    while True:
        try:
            ciclo += 1
            log.info(f"\n── Ciclo #{ciclo} ──────────────────────────────")

            for symbol in STOCKS:
                procesar_activo(symbol, es_crypto=False)
                time.sleep(1)  # Pausa breve entre llamadas a la API

            for symbol in CRYPTO:
                procesar_activo(symbol, es_crypto=True)
                time.sleep(1)

            # Mostrar resumen cada 5 ciclos
            if ciclo % 5 == 0:
                mostrar_resumen()

            log.info(f"⏳ Esperando {CHECK_INTERVAL}s para el próximo ciclo...")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log.info("\n🛑 Bot detenido manualmente.")
            mostrar_resumen()
            break
        except Exception as e:
            log.error(f"Error inesperado en el ciclo principal: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
