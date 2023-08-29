# QGAN-RSA

data preparing process:
    needed:
        poi data in scope;
        tiles imagery;
        osm data in scope;
    process:
        1. generate (quad tree) using (poi data);
        2. use (quad tree) tiles to cut (imagery);
        3. generate (hetero graph) using (osm data) and (quad tree);
        4. generate (imagery) embeddings with autoencoder;
