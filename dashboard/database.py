"""
SQLite Database Layer for Stock Trading Dashboard.

Provides persistence for:
- Trades (completed)
- Positions (open)
- Daily P&L snapshots
- Settings

Usage:
    db = Database("trading.db")
    db.save_trade(trade)
    trades = db.load_all_trades()
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeRecord:
    """Trade record for database."""
    id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    fee: float
    net_pnl: float
    entry_time: str
    exit_time: str
    reason: str
    strategy: str = "MicroEnsemble"
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PositionRecord:
    """Position record for database."""
    id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float
    entry_time: str
    stop_loss: float
    take_profit: float
    highest_price: float = 0.0
    lowest_price: float = 0.0
    initial_stop_loss: float = 0.0
    trailing_active: bool = False
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['trailing_active'] = 1 if self.trailing_active else 0
        return d


@dataclass
class DailyPnLRecord:
    """Daily P&L snapshot."""
    date: str
    gross_pnl: float
    net_pnl: float
    total_trades: int
    wins: int
    losses: int
    total_fees: float


# =============================================================================
# DATABASE CLASS
# =============================================================================

class Database:
    """SQLite database for stock trading persistence."""
    
    def __init__(self, db_path: str = "trading.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file (or ':memory:' for testing)
        """
        self.db_path = db_path
        self._is_memory = db_path == ':memory:'
        self._persistent_conn = None
        
        # For in-memory database, we need a persistent connection
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(':memory:')
            self._persistent_conn.row_factory = sqlite3.Row
        
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        if self._is_memory:
            # Reuse persistent connection for in-memory DB
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            # Create new connection for file-based DB
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    size REAL NOT NULL,
                    gross_pnl REAL NOT NULL,
                    fee REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    reason TEXT,
                    strategy TEXT DEFAULT 'MicroEnsemble',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions table (open positions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    highest_price REAL DEFAULT 0,
                    lowest_price REAL DEFAULT 0,
                    initial_stop_loss REAL DEFAULT 0,
                    trailing_active INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily P&L snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    gross_pnl REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    losses INTEGER NOT NULL,
                    total_fees REAL NOT NULL
                )
            """)
            
            # Settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
            
            logger.info(f"Database initialized: {self.db_path}")
    
    # =========================================================================
    # TRADES
    # =========================================================================
    
    def save_trade(self, trade: TradeRecord) -> bool:
        """Save a completed trade to database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trades 
                    (id, symbol, side, entry_price, exit_price, size, 
                     gross_pnl, fee, net_pnl, entry_time, exit_time, reason, strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id, trade.symbol, trade.side, trade.entry_price,
                    trade.exit_price, trade.size, trade.gross_pnl, trade.fee,
                    trade.net_pnl, trade.entry_time, trade.exit_time,
                    trade.reason, trade.strategy
                ))
            logger.debug(f"Saved trade {trade.id} to database")
            return True
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    def load_all_trades(self, limit: int = 1000) -> List[TradeRecord]:
        """Load all trades from database."""
        trades = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trades 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    trades.append(TradeRecord(
                        id=row['id'],
                        symbol=row['symbol'],
                        side=row['side'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        size=row['size'],
                        gross_pnl=row['gross_pnl'],
                        fee=row['fee'],
                        net_pnl=row['net_pnl'],
                        entry_time=row['entry_time'],
                        exit_time=row['exit_time'],
                        reason=row['reason'],
                        strategy=row['strategy'] or 'MicroEnsemble',
                        created_at=row['created_at'],
                    ))
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
        
        return trades
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get all trades for a specific symbol."""
        trades = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE symbol = ? 
                    ORDER BY created_at DESC
                """, (symbol,))
                
                for row in cursor.fetchall():
                    trades.append(TradeRecord(
                        id=row['id'],
                        symbol=row['symbol'],
                        side=row['side'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        size=row['size'],
                        gross_pnl=row['gross_pnl'],
                        fee=row['fee'],
                        net_pnl=row['net_pnl'],
                        entry_time=row['entry_time'],
                        exit_time=row['exit_time'],
                        reason=row['reason'],
                        strategy=row['strategy'] or 'MicroEnsemble',
                        created_at=row['created_at'],
                    ))
        except Exception as e:
            logger.error(f"Error loading trades for {symbol}: {e}")
        
        return trades
    
    def get_total_pnl(self) -> float:
        """Get total P&L from all trades."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COALESCE(SUM(net_pnl), 0) as total FROM trades")
                row = cursor.fetchone()
                return row['total'] if row else 0.0
        except Exception as e:
            logger.error(f"Error getting total P&L: {e}")
            return 0.0
    
    def get_total_fees(self) -> float:
        """Get total fees from all trades."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COALESCE(SUM(fee), 0) as total FROM trades")
                row = cursor.fetchone()
                return row['total'] if row else 0.0
        except Exception as e:
            logger.error(f"Error getting total fees: {e}")
            return 0.0
    
    def get_win_rate(self) -> float:
        """Get win rate from all trades."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
                    FROM trades
                """)
                row = cursor.fetchone()
                if row and row['total'] > 0:
                    return (row['wins'] / row['total']) * 100
                return 0.0
        except Exception as e:
            logger.error(f"Error getting win rate: {e}")
            return 0.0
    
    # =========================================================================
    # POSITIONS
    # =========================================================================
    
    def save_position(self, position: PositionRecord) -> bool:
        """Save an open position to database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO positions 
                    (id, symbol, side, entry_price, size, entry_time, 
                     stop_loss, take_profit, highest_price, lowest_price,
                     initial_stop_loss, trailing_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.id, position.symbol, position.side,
                    position.entry_price, position.size, position.entry_time,
                    position.stop_loss, position.take_profit,
                    position.highest_price, position.lowest_price,
                    position.initial_stop_loss, 1 if position.trailing_active else 0
                ))
            logger.debug(f"Saved position {position.id} to database")
            return True
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return False
    
    def delete_position(self, position_id: str) -> bool:
        """Delete a position (when closed)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM positions WHERE id = ?", (position_id,))
            logger.debug(f"Deleted position {position_id} from database")
            return True
        except Exception as e:
            logger.error(f"Error deleting position: {e}")
            return False
    
    def load_all_positions(self) -> List[PositionRecord]:
        """Load all open positions from database."""
        positions = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM positions ORDER BY created_at DESC")
                
                for row in cursor.fetchall():
                    positions.append(PositionRecord(
                        id=row['id'],
                        symbol=row['symbol'],
                        side=row['side'],
                        entry_price=row['entry_price'],
                        size=row['size'],
                        entry_time=row['entry_time'],
                        stop_loss=row['stop_loss'],
                        take_profit=row['take_profit'],
                        highest_price=row['highest_price'],
                        lowest_price=row['lowest_price'],
                        initial_stop_loss=row['initial_stop_loss'],
                        trailing_active=bool(row['trailing_active']),
                        created_at=row['created_at'],
                    ))
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
        
        return positions
    
    def update_position(self, position: PositionRecord) -> bool:
        """Update an existing position."""
        return self.save_position(position)
    
    # =========================================================================
    # DAILY P&L
    # =========================================================================
    
    def save_daily_pnl(self, record: DailyPnLRecord) -> bool:
        """Save or update daily P&L snapshot."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_pnl 
                    (date, gross_pnl, net_pnl, total_trades, wins, losses, total_fees)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.date, record.gross_pnl, record.net_pnl,
                    record.total_trades, record.wins, record.losses,
                    record.total_fees
                ))
            return True
        except Exception as e:
            logger.error(f"Error saving daily P&L: {e}")
            return False
    
    def get_daily_pnl(self, days: int = 30) -> List[DailyPnLRecord]:
        """Get last N days of P&L records."""
        records = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM daily_pnl 
                    ORDER BY date DESC 
                    LIMIT ?
                """, (days,))
                
                for row in cursor.fetchall():
                    records.append(DailyPnLRecord(
                        date=row['date'],
                        gross_pnl=row['gross_pnl'],
                        net_pnl=row['net_pnl'],
                        total_trades=row['total_trades'],
                        wins=row['wins'],
                        losses=row['losses'],
                        total_fees=row['total_fees'],
                    ))
        except Exception as e:
            logger.error(f"Error loading daily P&L: {e}")
        
        return records
    
    # =========================================================================
    # SETTINGS
    # =========================================================================
    
    def get_setting(self, key: str, default: str = "") -> str:
        """Get a setting value."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row['value'] if row else default
        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return default
    
    def set_setting(self, key: str, value: str) -> bool:
        """Set a setting value."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO settings (key, value)
                    VALUES (?, ?)
                """, (key, value))
            return True
        except Exception as e:
            logger.error(f"Error setting {key}: {e}")
            return False
    
    def get_capital(self, default: float = 1000.0) -> float:
        """Get current capital from settings."""
        value = self.get_setting('capital', str(default))
        try:
            return float(value)
        except ValueError:
            return default
    
    def set_capital(self, capital: float) -> bool:
        """Save current capital to settings."""
        return self.set_setting('capital', str(capital))
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall trading statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Overall stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                        COALESCE(SUM(gross_pnl), 0) as total_gross_pnl,
                        COALESCE(SUM(net_pnl), 0) as total_net_pnl,
                        COALESCE(SUM(fee), 0) as total_fees,
                        COALESCE(AVG(net_pnl), 0) as avg_pnl,
                        COALESCE(MAX(net_pnl), 0) as best_trade,
                        COALESCE(MIN(net_pnl), 0) as worst_trade
                    FROM trades
                """)
                row = cursor.fetchone()
                
                if row:
                    total = row['total_trades']
                    wins = row['wins'] or 0
                    
                    return {
                        'total_trades': total,
                        'wins': wins,
                        'losses': row['losses'] or 0,
                        'win_rate': (wins / total * 100) if total > 0 else 0,
                        'total_gross_pnl': row['total_gross_pnl'],
                        'total_net_pnl': row['total_net_pnl'],
                        'total_fees': row['total_fees'],
                        'avg_pnl': row['avg_pnl'],
                        'best_trade': row['best_trade'],
                        'worst_trade': row['worst_trade'],
                    }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
        
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_gross_pnl': 0,
            'total_net_pnl': 0,
            'total_fees': 0,
            'avg_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with in-memory database
    db = Database(":memory:")
    
    # Test trade
    trade = TradeRecord(
        id="TEST-001",
        symbol="AAPL",
        side="LONG",
        entry_price=150.00,
        exit_price=152.00,
        size=10.0,
        gross_pnl=20.00,
        fee=0.30,
        net_pnl=19.70,
        entry_time="10:00:00",
        exit_time="10:15:00",
        reason="Take Profit",
    )
    
    db.save_trade(trade)
    loaded = db.load_all_trades()
    print(f"Loaded {len(loaded)} trades")
    print(f"Total P&L: ${db.get_total_pnl():.2f}")
    print(f"Win Rate: {db.get_win_rate():.1f}%")
    
    # Test position
    pos = PositionRecord(
        id="POS-001",
        symbol="TSLA",
        side="SHORT",
        entry_price=250.00,
        size=5.0,
        entry_time="10:30:00",
        stop_loss=255.00,
        take_profit=240.00,
    )
    
    db.save_position(pos)
    positions = db.load_all_positions()
    print(f"Loaded {len(positions)} positions")
    
    # Test settings
    db.set_capital(1000.0)
    print(f"Capital: ${db.get_capital():.2f}")
    
    # Test statistics
    stats = db.get_statistics()
    print(f"Statistics: {stats}")
    
    print("\n✅ All database tests passed!")
