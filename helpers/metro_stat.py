# Recreate the full list of Athens metro stations (from previous message)
import pandas as pdd

athens_metro_all = [
    {"station": "Agia Marina", "latitude": 37.9904, "longitude": 23.6816},
    {"station": "Agia Paraskevi", "latitude": 38.0168, "longitude": 23.8216},
    {"station": "Agia Varvara", "latitude": 37.9900, "longitude": 23.6680},
    {"station": "Agios Antonios", "latitude": 38.0135, "longitude": 23.7050},
    {"station": "Agios Dimitrios", "latitude": 37.9301, "longitude": 23.7387},
    {"station": "Agios Eleftherios", "latitude": 38.0161, "longitude": 23.7368},
    {"station": "Agios Ioannis", "latitude": 37.9509, "longitude": 23.7366},
    {"station": "Agios Nikolaos", "latitude": 38.0014, "longitude": 23.7331},
    {"station": "Akropoli", "latitude": 37.9687, "longitude": 23.7282},
    {"station": "Alimos", "latitude": 37.9120, "longitude": 23.7235},
    {"station": "Ampelokipoi", "latitude": 37.9870, "longitude": 23.7577},
    {"station": "Ano Patisia", "latitude": 38.0165, "longitude": 23.7380},
    {"station": "Anthoupoli", "latitude": 38.0360, "longitude": 23.6910},
    {"station": "Argyroupoli", "latitude": 37.9050, "longitude": 23.7520},
    {"station": "Attiki", "latitude": 38.0026, "longitude": 23.7214},
    {"station": "Dafni", "latitude": 37.9411, "longitude": 23.7390},
    {"station": "Dimotiko Theatro", "latitude": 37.9420, "longitude": 23.6490},
    {"station": "Doukissis Plakentias", "latitude": 38.0169, "longitude": 23.8331},
    {"station": "Egaleo", "latitude": 37.9904, "longitude": 23.6816},
    {"station": "Eleonas", "latitude": 37.9883, "longitude": 23.7034},
    {"station": "Elliniko", "latitude": 37.8788, "longitude": 23.7486},
    {"station": "Ethniki Amyna", "latitude": 37.9923, "longitude": 23.7925},
    {"station": "Evangelismos", "latitude": 37.9763, "longitude": 23.7461},
    {"station": "Halandri", "latitude": 38.0170, "longitude": 23.7990},
    {"station": "Holargos", "latitude": 37.9947, "longitude": 23.7903},
    {"station": "Ilioupoli", "latitude": 37.9300, "longitude": 23.7520},
    {"station": "Iraklio", "latitude": 38.0520, "longitude": 23.7650},
    {"station": "Irini", "latitude": 38.0450, "longitude": 23.7840},
    {"station": "Kallithea", "latitude": 37.9505, "longitude": 23.7009},
    {"station": "KAT", "latitude": 38.0595, "longitude": 23.8057},
    {"station": "Katehaki", "latitude": 37.9875, "longitude": 23.7847},
    {"station": "Kato Patisia", "latitude": 38.0000, "longitude": 23.7330},
    {"station": "Kerameikos", "latitude": 37.9781, "longitude": 23.7116},
    {"station": "Kifisia", "latitude": 38.0728, "longitude": 23.8142},
    {"station": "Koropi", "latitude": 37.9160, "longitude": 23.8710},
    {"station": "Korydallos", "latitude": 37.9500, "longitude": 23.6470},
    {"station": "Larissa Station", "latitude": 38.0000, "longitude": 23.7320},
    {"station": "Maniatika", "latitude": 37.9450, "longitude": 23.6430},
    {"station": "Marousi", "latitude": 38.0560, "longitude": 23.8080},
    {"station": "Megaro Mousikis", "latitude": 37.9794, "longitude": 23.7521},
    {"station": "Metaxourgio", "latitude": 37.9850, "longitude": 23.7207},
    {"station": "Monastiraki", "latitude": 37.9763, "longitude": 23.7257},
    {"station": "Moschato", "latitude": 37.9530, "longitude": 23.6820},
    {"station": "Neratziotissa", "latitude": 38.0560, "longitude": 23.7920},
    {"station": "Nea Ionia", "latitude": 38.0350, "longitude": 23.7570},
    {"station": "Nea Filadelfia", "latitude": 38.0360, "longitude": 23.7380},
    {"station": "Neos Kosmos", "latitude": 37.9569, "longitude": 23.7287},
    {"station": "Nikaia", "latitude": 37.9500, "longitude": 23.6470},
    {"station": "Nomismatokopio", "latitude": 38.0080, "longitude": 23.7990},
    {"station": "Omonia", "latitude": 37.9848, "longitude": 23.7275},
    {"station": "Pallini", "latitude": 38.0030, "longitude": 23.8580},
    {"station": "Panepistimio", "latitude": 37.9832, "longitude": 23.7326},
    {"station": "Panormou", "latitude": 37.9890, "longitude": 23.7620},
    {"station": "Peania-Kantza", "latitude": 37.9450, "longitude": 23.8540},
    {"station": "Perissos", "latitude": 38.0330, "longitude": 23.7570},
    {"station": "Peristeri", "latitude": 38.0150, "longitude": 23.7000},
    {"station": "Petralona", "latitude": 37.9627, "longitude": 23.7083},
    {"station": "Piraeus", "latitude": 37.9470, "longitude": 23.6420},
    {"station": "Sepolia", "latitude": 38.0000, "longitude": 23.7130},
    {"station": "Sygrou-Fix", "latitude": 37.9623, "longitude": 23.7261},
    {"station": "Syntagma", "latitude": 37.9756, "longitude": 23.7345},
    {"station": "Tavros", "latitude": 37.9458, "longitude": 23.6933},
    {"station": "Thissio", "latitude": 37.9750, "longitude": 23.7192},
    {"station": "Victoria", "latitude": 37.9955, "longitude": 23.7281}
]

# Convert to DataFrame
athens_metro_all_df = pdd.DataFrame(athens_metro_all)

# Save to CSV
all_stations_path = "/mnt/data/Athens_Metro_Stations.csv"
athens_metro_all_df.to_csv(all_stations_path, index=False)


