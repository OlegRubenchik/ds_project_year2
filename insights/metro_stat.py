import pandas as pd

# Creating the DataFrame
data = {
    "Station": [
        "Acropoli", "Kerameikos", "Monastiraki", "Thiseio", "Panepistimio", "Attica", "Agios Antonios", 
        "Syntagma", "Biktṓria", "Tavros", "St. Peristeri", "Omonia", "Sepolia", "Syntagma", "Petralona", 
        "Dafni", "Eleonas", "Kallithea", "Holargos", "Perissós"
    ],
    "Latitude": [
        37.96877, 37.9785433, 37.9760854, 37.9767093, 37.9803782, 37.9992951, 38.0066611,
        37.974647, 37.9930561, 37.9624659, 38.0131616, 37.984183, 38.00266999999999, 37.9755845,
        37.9686198, 37.9492026, 37.98789379999999, 37.9604041, 38.00451899999999, 38.032728
    ],
    "Longitude": [
        23.72962, 23.7115372, 23.7256256, 23.7207061, 23.7330586, 23.7220981, 23.699481,
        23.7356793, 23.7304093, 23.7033906, 23.6955017, 23.728692, 23.7136, 23.7353853,
        23.7092682, 23.7372233, 23.6941769, 23.6973321, 23.794723, 23.74467
    ]
}

athens_metro_all = pd.DataFrame(data)
print(athens_metro_all)

