def prepare(param):
    param['zone'] = param['loc'].map(state_to_zone)
    param['title'] = param['title'].map(house_type_ranks)

    category_frequencies = param['loc'].value_counts(normalize=True)
    loc_frequency_mapping = category_frequencies.to_dict()
    param['loc'] = param['loc'].map(loc_frequency_mapping)

    param['rooms'] = param['bathroom'] + param['bedroom']
    param['bathroom_ratio'] = param['bathroom']/(param['bathroom'] + param['bedroom'])

    param['zone'] = param['zone'].astype('category').cat.codes

    return param


house_type_ranks = {
    'Cottage': 1,
    'Bungalow': 2,
    'Townhouse': 3,
    'Terrace duplex': 4,
    'Detached duplex': 5,
    'Semi-detached duplex': 6,
    'Flat': 7,
    'Penthouse': 8,
    'Apartment': 9,
    'Mansion': 10
}

state_to_zone = {
    "Abia": "South-East",
    "Adamawa": "North-East",
    "Akwa Ibom": "South-South",
    "Anambra": "South-East",
    "Bauchi": "North-East",
    "Bayelsa": "South-South",
    "Benue": "North-Central",
    "Borno": "North-East",
    "Cross River": "South-South",
    "Delta": "South-South",
    "Ebonyi": "South-East",
    "Edo": "South-South",
    "Ekiti": "South-West",
    "Enugu": "South-East",
    "Gombe": "North-East",
    "Imo": "South-East",
    "Jigawa": "North-West",
    "Kaduna": "North-West",
    "Kano": "North-West",
    "Katsina": "North-West",
    "Kebbi": "North-West",
    "Kogi": "North-Central",
    "Kwara": "North-Central",
    "Lagos": "South-West",
    "Nasarawa": "North-Central",
    "Niger": "North-Central",
    "Ogun": "South-West",
    "Ondo": "South-West",
    "Osun": "South-West",
    "Oyo": "South-West",
    "Plateau": "North-Central",
    "Rivers": "South-South",
    "Sokoto": "North-West",
    "Taraba": "North-East",
    "Yobe": "North-East",
    "Zamfara": "North-West",
}