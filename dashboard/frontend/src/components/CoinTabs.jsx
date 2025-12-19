import { useDispatch } from 'react-redux';
import { useState } from 'react';
import { Power, TrendingUp } from 'lucide-react';
import { setActiveSymbol } from '../store/tradingSlice';

// Top crypto coins for dropdown
const TOP_COINS = [
    { s: 'BTC', n: 'Bitcoin' }, { s: 'ETH', n: 'Ethereum' }, { s: 'SOL', n: 'Solana' },
    { s: 'BNB', n: 'Binance Coin' }, { s: 'XRP', n: 'Ripple' }, { s: 'ADA', n: 'Cardano' },
    { s: 'AVAX', n: 'Avalanche' }, { s: 'DOGE', n: 'Dogecoin' }, { s: 'DOT', n: 'Polkadot' },
    { s: 'TRX', n: 'Tron' }, { s: 'MATIC', n: 'Polygon' }, { s: 'ATOM', n: 'Cosmos' },
    { s: 'LTC', n: 'Litecoin' }, { s: 'SHIB', n: 'Shiba Inu' }, { s: 'UNI', n: 'Uniswap' },
    { s: 'LINK', n: 'Chainlink' }, { s: 'XLM', n: 'Stellar' }, { s: 'ETC', n: 'Ethereum Classic' },
    { s: 'BCH', n: 'Bitcoin Cash' }, { s: 'XMR', n: 'Monero' }, { s: 'ALGO', n: 'Algorand' },
    { s: 'NEAR', n: 'Near Protocol' }, { s: 'FIL', n: 'Filecoin' }, { s: 'VET', n: 'VeChain' },
    { s: 'ICP', n: 'Internet Computer' }, { s: 'APE', n: 'ApeCoin' }, { s: 'SAND', n: 'Sandbox' },
    { s: 'MANA', n: 'Decentraland' }, { s: 'QNT', n: 'Quant' }, { s: 'THETA', n: 'Theta' },
    { s: 'AAVE', n: 'Aave' }, { s: 'AXS', n: 'Axie Infinity' }, { s: 'EOS', n: 'EOS' },
    { s: 'EGLD', n: 'MultiversX' }, { s: 'XTZ', n: 'Tezos' }, { s: 'HBAR', n: 'Hedera' },
    { s: 'GRT', n: 'The Graph' }, { s: 'MKR', n: 'Maker' }, { s: 'BSV', n: 'Bitcoin SV' },
    { s: 'FTM', n: 'Fantom' }, { s: 'RUNE', n: 'Thorchain' }, { s: 'ZEC', n: 'Zcash' },
    { s: 'SNX', n: 'Synthetix' }, { s: 'NEO', n: 'Neo' }, { s: 'FLOW', n: 'Flow' },
    { s: 'CHZ', n: 'Chiliz' }, { s: 'CRV', n: 'Curve' }, { s: 'IOTA', n: 'IOTA' },
    { s: 'ENJ', n: 'Enjin' }, { s: 'BAT', n: 'Basic Attention Token' },
    { s: 'LRC', n: 'Loopring' }, { s: 'KSM', n: 'Kusama' }, { s: 'ZIL', n: 'Zilliqa' },
    { s: 'STX', n: 'Stacks' }, { s: 'COMP', n: 'Compound' }, { s: 'GMT', n: 'STEPN' },
    { s: 'WAVES', n: 'Waves' }, { s: 'KAVA', n: 'Kava' }, { s: 'CELO', n: 'Celo' },
    { s: '1INCH', n: '1inch' }, { s: 'HOT', n: 'Holo' }, { s: 'OMG', n: 'OMG Network' },
    { s: 'ICX', n: 'Icon' }, { s: 'QTUM', n: 'Qtum' }, { s: 'IOST', n: 'IOST' },
    { s: 'RVN', n: 'Ravencoin' }, { s: 'ONT', n: 'Ontology' }, { s: 'KNC', n: 'Kyber' },
    { s: 'ZRX', n: '0x' }, { s: 'GALA', n: 'Gala' }, { s: 'WAXP', n: 'Wax' },
    { s: 'ANKR', n: 'Ankr' }, { s: 'SC', n: 'Siacoin' }, { s: 'HIVE', n: 'Hive' },
    { s: 'SUSHI', n: 'SushiSwap' }, { s: 'YFI', n: 'Yearn.finance' }, { s: 'UMA', n: 'UMA' },
    { s: 'SKL', n: 'Skale' }, { s: 'SRM', n: 'Serum' }, { s: 'REN', n: 'Ren' },
    { s: 'DGB', n: 'DigiByte' }, { s: 'OCEAN', n: 'Ocean Protocol' }, { s: 'STORJ', n: 'Storj' },
    { s: 'GLM', n: 'Golem' }, { s: 'LSK', n: 'Lisk' }, { s: 'BAND', n: 'Band Protocol' },
    { s: 'NANO', n: 'Nano' }
].sort((a, b) => a.s.localeCompare(b.s));

// Top US stocks for dropdown
const TOP_STOCKS = [
    { s: 'AAPL', n: 'Apple' }, { s: 'MSFT', n: 'Microsoft' }, { s: 'GOOGL', n: 'Alphabet' },
    { s: 'AMZN', n: 'Amazon' }, { s: 'NVDA', n: 'NVIDIA' }, { s: 'META', n: 'Meta' },
    { s: 'TSLA', n: 'Tesla' }, { s: 'BRK.B', n: 'Berkshire' }, { s: 'JPM', n: 'JPMorgan' },
    { s: 'V', n: 'Visa' }, { s: 'UNH', n: 'UnitedHealth' }, { s: 'JNJ', n: 'Johnson & J' },
    { s: 'WMT', n: 'Walmart' }, { s: 'MA', n: 'Mastercard' }, { s: 'PG', n: 'P&G' },
    { s: 'HD', n: 'Home Depot' }, { s: 'BAC', n: 'Bank of America' }, { s: 'XOM', n: 'Exxon' },
    { s: 'DIS', n: 'Disney' }, { s: 'NFLX', n: 'Netflix' }, { s: 'INTC', n: 'Intel' },
    { s: 'AMD', n: 'AMD' }, { s: 'ADBE', n: 'Adobe' }, { s: 'CRM', n: 'Salesforce' },
    { s: 'PYPL', n: 'PayPal' }, { s: 'CSCO', n: 'Cisco' }, { s: 'CMCSA', n: 'Comcast' },
    { s: 'PEP', n: 'PepsiCo' }, { s: 'KO', n: 'Coca-Cola' }, { s: 'NKE', n: 'Nike' },
    { s: 'MCD', n: "McDonald's" }, { s: 'T', n: 'AT&T' }, { s: 'VZ', n: 'Verizon' },
    { s: 'COST', n: 'Costco' }, { s: 'SBUX', n: 'Starbucks' }, { s: 'BA', n: 'Boeing' },
    { s: 'GE', n: 'General Electric' }, { s: 'F', n: 'Ford' }, { s: 'GM', n: 'GM' },
    { s: 'UBER', n: 'Uber' }, { s: 'LYFT', n: 'Lyft' }, { s: 'SQ', n: 'Block' },
    { s: 'COIN', n: 'Coinbase' }, { s: 'HOOD', n: 'Robinhood' }, { s: 'PLTR', n: 'Palantir' },
    { s: 'SNOW', n: 'Snowflake' }, { s: 'RBLX', n: 'Roblox' }, { s: 'RIVN', n: 'Rivian' },
    { s: 'LCID', n: 'Lucid' }, { s: 'SPCE', n: 'Virgin Galactic' },
].sort((a, b) => a.s.localeCompare(b.s));

// Stock color palette for visual differentiation
const STOCK_COLORS = [
    'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-rose-500',
    'bg-amber-500', 'bg-cyan-500', 'bg-indigo-500', 'bg-emerald-500',
];

const getStockColor = (symbol) => {
    const hash = symbol.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
    return STOCK_COLORS[hash % STOCK_COLORS.length];
};

export const CoinTabs = ({ coins, activeSymbol, sendMessage, data, marketType = 'crypto' }) => {
    const dispatch = useDispatch();
    const [isAdding, setIsAdding] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    const isStocks = marketType === 'stocks';
    const assetList = isStocks ? TOP_STOCKS : TOP_COINS;
    const symbolSuffix = isStocks ? '' : 'USDT';

    const handleSelect = (symbol) => {
        dispatch(setActiveSymbol(symbol));
        sendMessage({ action: 'set_active', symbol });
    };

    const handleAddAsset = (symbol) => {
        const action = isStocks ? 'add_stock' : 'add_coin';
        const fullSymbol = isStocks ? symbol : symbol + 'USDT';
        sendMessage({ action, symbol: fullSymbol });
        setSearchQuery('');
        setIsAdding(false);
    };

    // Filter available assets (exclude already added ones)
    const filteredAssets = assetList
        .filter(c => !coins.includes(isStocks ? c.s : c.s + 'USDT'))
        .filter(c => c.s.toLowerCase().includes(searchQuery.toLowerCase()) || c.n.toLowerCase().includes(searchQuery.toLowerCase()));

    // Loader state
    const [pendingAsset, setPendingAsset] = useState(null);

    // Watch for pending asset to arrive
    const expectedSymbol = pendingAsset ? (isStocks ? pendingAsset : pendingAsset + 'USDT') : null;
    if (pendingAsset && coins.includes(expectedSymbol)) {
        setPendingAsset(null);
        handleSelect(expectedSymbol);
    }

    const handleAddAssetAndLoad = (symbol) => {
        setPendingAsset(symbol);
        handleAddAsset(symbol);
    };

    return (
        <>
            {/* Loading Overlay */}
            {pendingAsset && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
                    <div className="bg-bg-card border border-border rounded-xl p-8 flex flex-col items-center gap-4 shadow-2xl max-w-sm w-full mx-4">
                        <div className="relative w-16 h-16">
                            <div className="absolute inset-0 border-4 border-gray-700/50 rounded-full"></div>
                            <div className={`absolute inset-0 border-4 border-t-${isStocks ? 'green' : 'emerald'}-500 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin`}></div>
                            {isStocks ? (
                                <div className={`absolute inset-3 w-10 h-10 rounded-full ${getStockColor(pendingAsset)} flex items-center justify-center`}>
                                    <span className="text-white font-bold text-xs">{pendingAsset.slice(0, 2)}</span>
                                </div>
                            ) : (
                                <img
                                    src={`https://assets.coincap.io/assets/icons/${pendingAsset.toLowerCase()}@2x.png`}
                                    alt={pendingAsset}
                                    className="absolute inset-4 w-8 h-8 rounded-full"
                                    onError={(e) => { e.target.style.display = 'none' }}
                                />
                            )}
                        </div>

                        <div className="text-center">
                            <h3 className="text-xl font-bold text-white mb-1">Initializing {pendingAsset}</h3>
                            <p className="text-gray-400 text-sm">
                                {isStocks ? (
                                    <>
                                        Fetching Yahoo Finance data...<br />
                                        Building 5m candles...<br />
                                        Calculating technicals...
                                    </>
                                ) : (
                                    <>
                                        Establishing WebSocket feeds...<br />
                                        Fetching 1s candles...<br />
                                        Calculating technicals...
                                    </>
                                )}
                            </p>
                        </div>

                        <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2 overflow-hidden">
                            <div className={`${isStocks ? 'bg-green-500' : 'bg-emerald-500'} h-full w-1/3 animate-[shimmer_1s_infinite_linear]`} style={{ width: '60%' }}></div>
                        </div>
                    </div>
                </div>
            )}

            <div className="flex items-center gap-2 bg-bg-secondary p-1 rounded-lg border border-border relative flex-wrap">
                {coins.map((symbol) => {
                    const coinData = data?.coins?.[symbol] || {};
                    const autoTradeEnabled = coinData.auto_trade_enabled !== false;
                    const baseSymbol = isStocks ? symbol : symbol.replace('USDT', '').toLowerCase();

                    return (
                        <div key={symbol} className="flex items-center gap-1">
                            <button
                                onClick={() => handleSelect(symbol)}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${activeSymbol === symbol
                                    ? 'bg-bg-card text-white shadow-lg border border-gray-600'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                {isStocks ? (
                                    <div className={`w-5 h-5 rounded-full ${getStockColor(symbol)} flex items-center justify-center`}>
                                        <span className="text-white font-bold text-[8px]">{symbol.slice(0, 2)}</span>
                                    </div>
                                ) : (
                                    <img
                                        src={`https://assets.coincap.io/assets/icons/${baseSymbol}@2x.png`}
                                        alt={baseSymbol}
                                        className="w-4 h-4 rounded-full"
                                        onError={(e) => { e.target.style.display = 'none' }}
                                    />
                                )}
                                {isStocks ? symbol : symbol.replace('USDT', '')}
                            </button>

                            {/* Auto-trade toggle */}
                            <button
                                onClick={() => sendMessage({ action: 'toggle_auto_trade', symbol })}
                                className={`p-1.5 rounded-md transition-all ${autoTradeEnabled
                                    ? `${isStocks ? 'bg-green-600/20 text-green-400 hover:bg-green-600/30' : 'bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/30'}`
                                    : 'bg-gray-700 text-gray-500 hover:bg-gray-600'
                                    }`}
                                title={`Auto-trade ${autoTradeEnabled ? 'ON' : 'OFF'}`}
                            >
                                <Power className="w-3 h-3" />
                            </button>
                        </div>
                    );
                })}

                {/* Add Asset Dropdown */}
                <div className="relative">
                    <button
                        onClick={() => setIsAdding(!isAdding)}
                        className={`px-3 py-1.5 text-gray-500 hover:text-gray-300 text-xs hover:bg-white/5 rounded transition-all ${isAdding ? 'text-white bg-white/10' : ''}`}
                    >
                        + ADD {isStocks ? 'STOCK' : 'COIN'}
                    </button>

                    {isAdding && (
                        <div className="absolute top-full right-0 mt-2 w-64 bg-bg-card border border-border rounded-lg shadow-xl z-50 overflow-hidden">
                            <div className="p-2 border-b border-border">
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder={isStocks ? "Search (e.g. Apple)..." : "Search (e.g. Cosmos)..."}
                                    className="w-full bg-bg-secondary text-xs px-2 py-1.5 rounded border border-border focus:outline-none focus:border-emerald-500"
                                    autoFocus
                                />
                            </div>
                            <div className="max-h-60 overflow-y-auto">
                                {filteredAssets.length > 0 ? (
                                    filteredAssets.map(asset => (
                                        <button
                                            key={asset.s}
                                            onClick={() => handleAddAssetAndLoad(asset.s)}
                                            className="w-full text-left px-3 py-2 text-sm text-gray-400 hover:bg-white/5 hover:text-white transition-colors flex justify-between items-center group"
                                        >
                                            <div className="flex items-center gap-2">
                                                {isStocks ? (
                                                    <div className={`w-6 h-6 rounded-full ${getStockColor(asset.s)} flex items-center justify-center opacity-70 group-hover:opacity-100`}>
                                                        <span className="text-white font-bold text-[8px]">{asset.s.slice(0, 2)}</span>
                                                    </div>
                                                ) : (
                                                    <img
                                                        src={`https://assets.coincap.io/assets/icons/${asset.s.toLowerCase()}@2x.png`}
                                                        alt={asset.s}
                                                        className="w-6 h-6 rounded-full opacity-70 group-hover:opacity-100"
                                                        onError={(e) => { e.target.style.display = 'none' }}
                                                    />
                                                )}
                                                <div className="flex flex-col">
                                                    <span className="font-bold leading-none">{asset.s}</span>
                                                    <span className="text-[10px] text-gray-600 group-hover:text-gray-400 leading-none mt-1">{asset.n}</span>
                                                </div>
                                            </div>
                                        </button>
                                    ))
                                ) : (
                                    <div className="px-3 py-2 text-xs text-gray-600 text-center">
                                        No {isStocks ? 'stocks' : 'coins'} found
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Overlay to close dropdown when clicking outside */}
                {isAdding && (
                    <div
                        className="fixed inset-0 z-40 bg-transparent"
                        onClick={() => setIsAdding(false)}
                    />
                )}
            </div>
        </>
    );
};
