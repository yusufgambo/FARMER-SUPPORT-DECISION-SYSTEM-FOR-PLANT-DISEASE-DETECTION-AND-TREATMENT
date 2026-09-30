import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Farmer Support Decision System for Plant Disease Detection and Treatment",
    page_icon="🌿",
    layout="centered"
)

# ── Load Disease Model ─────────────────────────
model_loaded = False
try:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(
        model_path="models/plant_disease_model_38.tflite"
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    pass

# ── Load Leaf Validator ────────────────────────
validator_loaded = False
try:
    from ai_edge_litert.interpreter import Interpreter as Val
    validator = Val(
        model_path="models/leaf_validator.tflite"
    )
    validator.allocate_tensors()
    val_input = validator.get_input_details()
    val_output = validator.get_output_details()
    validator_loaded = True
except Exception as e:
    pass

# ── AI Disease Classes ─────────────────────────
ai_classes = [
    "Apple Black Rot","Apple Cedar Rust","Apple Healthy","Apple Scab",
    "Blueberry Healthy","Cherry Healthy","Cherry Powdery Mildew",
    "Corn Cercospora Leaf Spot","Corn Common Rust","Corn Healthy",
    "Corn Northern Leaf Blight","Grape Black Measles","Grape Black Rot",
    "Grape Healthy","Grape Leaf Blight","Orange Citrus Greening",
    "Peach Bacterial Spot","Peach Healthy","Pepper Bacterial Spot",
    "Pepper Healthy","Potato Early Blight","Potato Healthy",
    "Potato Late Blight","Raspberry Healthy","Soybean Healthy",
    "Squash Powdery Mildew","Strawberry Healthy","Strawberry Leaf Scorch",
    "Tomato Bacterial Spot","Tomato Early Blight","Tomato Healthy",
    "Tomato Late Blight","Tomato Leaf Mold","Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot","Tomato Spider Mites",
    "Tomato Target Spot","Tomato Yellow Leaf Curl Virus"
]

# ── Disease Database ───────────────────────────
diseases_db = {
    "tomato bacterial spot":{"name":"Tomato Bacterial Spot","crop":"Tomato","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Bacterium: Xanthomonas vesicatoria","symptoms":"Small dark water-soaked spots on leaves turning brown with yellow halo.","treatment":["Apply copper-based bactericide every 7 days","Remove infected plant parts immediately","Avoid working with plants when wet","Use streptomycin spray in severe cases"],"prevention":["Use certified disease-free seeds","Avoid overhead irrigation","Rotate crops every season","Disinfect tools regularly"],"fertilizer":["Apply Calcium-rich fertilizer to strengthen cell walls","Use balanced NPK 20-20-20","Avoid excessive Nitrogen","Apply Potassium foliar spray to boost immunity"]},
    "tomato early blight":{"name":"Tomato Early Blight","crop":"Tomato","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Alternaria solani","symptoms":"Brown spots with yellow rings on lower leaves. Leaves turn yellow and drop.","treatment":["Remove infected leaves immediately","Spray Chlorothalonil fungicide every 7 days","Apply copper-based fungicide as alternative","Water at base only"],"prevention":["Rotate crops every season","Plant resistant varieties","Ensure proper spacing for air circulation","Remove plant debris after harvest"],"fertilizer":["Apply Potassium fertilizer to boost plant immunity","Use NPK 15-15-15 balanced fertilizer","Avoid excessive Nitrogen fertilizer","Apply Calcium and Magnesium foliar spray"]},
    "tomato late blight":{"name":"Tomato Late Blight","crop":"Tomato","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Water mold: Phytophthora infestans","symptoms":"Dark watery spots on leaves and stems. White mold visible underneath leaves.","treatment":["Remove and destroy infected plants immediately","Spray Mancozeb or Metalaxyl every 5-7 days","Do not compost — burn infected plants","Increase spray frequency during wet weather"],"prevention":["Use drip irrigation only","Plant certified disease-free seeds","Ensure good field drainage","Monitor crops during rainy season"],"fertilizer":["Apply Phosphorus fertilizer to strengthen roots","Use Potassium-rich fertilizer to boost immunity","Avoid high Nitrogen fertilizer","Apply Calcium nitrate to strengthen plant tissue"]},
    "tomato leaf mold":{"name":"Tomato Leaf Mold","crop":"Tomato","emoji":"🟡","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Passalora fulva","symptoms":"Yellow spots on upper leaf surface with olive-green mold on underside of leaves.","treatment":["Apply Chlorothalonil or Mancozeb fungicide","Improve air circulation around plants","Remove and destroy infected leaves","Reduce humidity around plants"],"prevention":["Plant resistant tomato varieties","Space plants properly for airflow","Avoid overhead watering","Maintain low humidity"],"fertilizer":["Apply balanced NPK fertilizer","Use Potassium fertilizer to boost resistance","Avoid excessive Nitrogen","Apply micronutrient foliar spray"]},
    "tomato mosaic virus":{"name":"Tomato Mosaic Virus","crop":"Tomato","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus: Tomato Mosaic Virus (ToMV)","symptoms":"Mosaic pattern of light and dark green on leaves, stunted growth, distorted fruits.","treatment":["Remove and destroy all infected plants","Control aphid vectors with insecticide","Disinfect all tools with bleach solution","Wash hands thoroughly before handling plants"],"prevention":["Use virus-free certified seeds only","Control insect vectors consistently","Remove infected plants immediately","Avoid growing tobacco near tomato plants"],"fertilizer":["Apply balanced NPK to maintain plant strength","Use Zinc and Boron foliar spray","Avoid excessive Nitrogen","Apply Potassium to boost overall plant immunity"]},
    "tomato septoria leaf spot":{"name":"Tomato Septoria Leaf Spot","crop":"Tomato","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Septoria lycopersici","symptoms":"Small circular spots with dark borders and light grey centers on leaves.","treatment":["Apply Chlorothalonil or copper fungicide","Remove infected lower leaves immediately","Spray every 7-10 days","Avoid wetting foliage when watering"],"prevention":["Rotate crops for minimum 2 years","Use mulch to prevent soil splash onto leaves","Space plants for good airflow","Remove all crop debris after harvest"],"fertilizer":["Apply Potassium fertilizer to boost resistance","Use balanced NPK 15-15-15","Avoid high Nitrogen fertilizer","Apply Calcium foliar spray"]},
    "tomato spider mites":{"name":"Tomato Spider Mites","crop":"Tomato","emoji":"🟡","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Pest: Tetranychus urticae","symptoms":"Yellow stippling on leaves, fine webbing on underside. Leaves turn bronze and drop.","treatment":["Apply miticide or insecticidal soap spray","Spray neem oil every 5 days","Increase humidity around plants","Remove heavily infested leaves immediately"],"prevention":["Monitor plants regularly for early detection","Avoid water stress which attracts mites","Remove infested leaves promptly","Use reflective mulch to repel mites"],"fertilizer":["Apply Silicon fertilizer to strengthen leaf tissue","Use balanced NPK fertilizer","Apply Potassium to boost plant immunity","Avoid excessive Nitrogen which promotes soft growth"]},
    "tomato target spot":{"name":"Tomato Target Spot","crop":"Tomato","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Corynespora cassiicola","symptoms":"Brown spots with concentric rings resembling a target on leaves and fruits.","treatment":["Apply Chlorothalonil or Mancozeb fungicide","Remove all infected plant material","Spray every 7-14 days","Improve air circulation around plants"],"prevention":["Use disease-free transplants only","Rotate crops regularly","Avoid overhead irrigation","Remove all crop debris after harvest"],"fertilizer":["Apply Potassium-rich fertilizer","Use NPK 15-15-15 balanced fertilizer","Apply Calcium and Boron foliar spray","Avoid excessive Nitrogen"]},
    "tomato yellow leaf curl virus":{"name":"Tomato Yellow Leaf Curl Virus","crop":"Tomato","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Begomovirus transmitted by whitefly Bemisia tabaci","symptoms":"Upward curling and yellowing of leaves, stunted plant growth, significantly reduced fruit production.","treatment":["Remove and destroy all infected plants","Apply insecticide to control whitefly population","Use yellow sticky traps to monitor whiteflies","No chemical cure exists for the virus itself"],"prevention":["Plant virus-resistant tomato varieties","Use insect-proof screens in greenhouse","Control whitefly with neem oil spray","Use reflective mulch to repel whiteflies"],"fertilizer":["Apply balanced NPK to maintain plant strength","Use Potassium fertilizer to boost immunity","Apply Zinc foliar spray","Avoid excessive Nitrogen fertilizer"]},
    "tomato healthy":{"name":"Tomato Healthy","crop":"Tomato","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good soil nutrition","Monitor plants regularly for early disease signs","Practice crop rotation","Maintain proper spacing and irrigation"],"fertilizer":["Apply balanced NPK 15-15-15 every 2 weeks","Use Calcium and Magnesium foliar spray","Apply Potassium fertilizer during fruiting stage","Use compost to improve soil health and fertility"]},
    "potato early blight":{"name":"Potato Early Blight","crop":"Potato","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Alternaria solani","symptoms":"Dark brown spots with concentric rings on older lower leaves.","treatment":["Apply Chlorothalonil or Mancozeb fungicide","Remove and destroy infected leaves","Spray every 7-10 days during season","Avoid overhead irrigation"],"prevention":["Use certified disease-free seed potatoes","Rotate crops every 2-3 years","Maintain proper plant nutrition","Remove all crop debris after harvest"],"fertilizer":["Apply Potassium fertilizer to boost immunity","Use NPK 15-15-15","Avoid excessive Nitrogen","Apply Calcium foliar spray"]},
    "potato late blight":{"name":"Potato Late Blight","crop":"Potato","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Water mold: Phytophthora infestans","symptoms":"Water-soaked spots on leaves turning dark brown. White mold on leaf undersides. Tubers rot rapidly.","treatment":["Apply Metalaxyl or Cymoxanil fungicide immediately","Remove all infected plant material","Harvest tubers early if disease is severe","Spray preventively during wet weather"],"prevention":["Plant certified resistant potato varieties","Avoid planting in poorly drained fields","Use certified disease-free seed potatoes","Monitor weather forecasts for blight conditions"],"fertilizer":["Apply Phosphorus to strengthen root system","Use Potassium-rich fertilizer","Avoid high Nitrogen fertilizer","Apply Calcium nitrate"]},
    "potato healthy":{"name":"Potato Healthy","crop":"Potato","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly","Use certified seed potatoes"],"fertilizer":["Apply NPK 15-15-15 fertilizer","Use Potassium during tuber formation","Apply Calcium and Magnesium","Use compost to improve soil"]},
    "corn common rust":{"name":"Corn Common Rust","crop":"Corn/Maize","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Puccinia sorghi","symptoms":"Small golden-brown to brick-red pustules scattered on both upper and lower leaf surfaces.","treatment":["Apply Propiconazole or Azoxystrobin fungicide","Spray at first sign of disease","Repeat application every 14 days","Remove and destroy heavily infected plants"],"prevention":["Plant rust-resistant corn varieties","Plant early to avoid peak rust season","Monitor fields regularly","Maintain good crop nutrition throughout season"],"fertilizer":["Apply Potassium fertilizer to boost resistance","Use NPK 15-15-15","Avoid excessive Nitrogen","Apply Silicon fertilizer to strengthen leaf tissue"]},
    "corn northern leaf blight":{"name":"Corn Northern Leaf Blight","crop":"Corn/Maize","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Exserohilum turcicum","symptoms":"Long tan to grey cigar-shaped lesions running along the length of leaves.","treatment":["Apply Propiconazole or Tebuconazole fungicide","Spray at tasseling stage for best results","Remove and destroy infected plant debris","Avoid dense planting to improve air circulation"],"prevention":["Plant resistant corn hybrids","Rotate crops with non-corn crops","Till soil to bury infected crop debris","Avoid excessive nitrogen fertilizer application"],"fertilizer":["Apply balanced NPK fertilizer","Use Potassium to boost immunity","Avoid excessive Nitrogen","Apply Zinc foliar spray"]},
    "corn cercospora leaf spot":{"name":"Corn Cercospora Leaf Spot","crop":"Corn/Maize","emoji":"🟡","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Cercospora zeae-maydis","symptoms":"Rectangular grey to tan lesions with dark brown borders running between leaf veins.","treatment":["Apply Strobilurin or Triazole fungicide","Spray at early stage of disease onset","Improve field drainage","Remove and destroy infected crop residue"],"prevention":["Plant resistant corn varieties","Rotate crops regularly","Reduce plant density for better airflow","Avoid minimum tillage in previously infected fields"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK 15-15-15","Apply Zinc and Boron foliar spray","Avoid excessive Nitrogen"]},
    "corn healthy":{"name":"Corn Healthy","crop":"Corn/Maize","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15 at planting","Use Urea for top dressing","Apply Zinc foliar spray","Use compost"]},
    "rice blast":{"name":"Rice Blast Disease","crop":"Rice","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungus: Magnaporthe oryzae","symptoms":"Diamond-shaped lesions with grey centers and dark brown borders on leaves and neck of panicle.","treatment":["Apply Tricyclazole or Isoprothiolane fungicide","Spray at booting stage and repeat 10 days later","Drain fields periodically to reduce humidity","Remove and destroy all infected plant debris"],"prevention":["Plant blast-resistant rice varieties","Avoid excessive nitrogen fertilization","Maintain proper water management in fields","Use certified disease-free seeds only"],"fertilizer":["Reduce Nitrogen fertilizer application immediately","Apply Silicon fertilizer to strengthen stems","Use Potassium to boost plant immunity","Apply balanced NPK before blast season starts"]},
    "rice brown spot":{"name":"Rice Brown Spot","crop":"Rice","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Cochliobolus miyabeanus","symptoms":"Oval brown spots with distinct yellow halo on leaves. Spots may also appear on grains.","treatment":["Apply Mancozeb or Iprodione fungicide","Spray at tillering stage","Improve soil fertility and nutrition","Remove infected plant debris"],"prevention":["Use certified seeds","Maintain proper plant nutrition","Avoid water stress","Rotate crops"],"fertilizer":["Apply balanced NPK fertilizer","Improve soil nutrition with compost","Use Zinc fertilizer to correct deficiency","Apply Potassium to boost resistance"]},
    "rice bacterial blight":{"name":"Rice Bacterial Blight","crop":"Rice","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Bacterium: Xanthomonas oryzae pv. oryzae","symptoms":"Water-soaked lesions on leaf margins that turn yellow then white and dry out.","treatment":["Apply copper-based bactericide","Drain fields and keep dry","Remove and destroy infected plants","Avoid excessive nitrogen fertilizer"],"prevention":["Plant resistant varieties","Use certified seeds","Avoid flood irrigation","Maintain field hygiene"],"fertilizer":["Reduce Nitrogen fertilizer immediately","Apply Potassium to boost immunity","Use Silicon fertilizer","Apply balanced NPK after recovery"]},
    "rice sheath blight":{"name":"Rice Sheath Blight","crop":"Rice","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Rhizoctonia solani","symptoms":"Oval to irregular lesions on leaf sheaths near water line with brown borders and grey centers.","treatment":["Apply Propiconazole or Hexaconazole","Spray at early tillering","Drain field to reduce humidity","Remove infected stubble"],"prevention":["Reduce plant density","Avoid excessive nitrogen","Use resistant varieties","Rotate crops"],"fertilizer":["Reduce Nitrogen application","Apply Silicon fertilizer","Use Potassium to strengthen stems","Apply balanced NPK"]},
    "rice tungro":{"name":"Rice Tungro Disease","crop":"Rice","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus complex transmitted by green leafhopper Nephotettix virescens","symptoms":"Yellow-orange discoloration of leaves starting from tips. Severely stunted plant growth and reduced tillering.","treatment":["Control leafhopper with systemic insecticide","Remove and destroy all infected plants","Use mineral oil spray to slow virus spread","No chemical cure exists for the virus"],"prevention":["Plant tungro-resistant rice varieties","Control leafhopper population","Synchronize planting dates with neighbors","Remove infected plants as early as possible"],"fertilizer":["Apply balanced NPK to maintain plant strength","Use Zinc foliar spray","Apply Potassium to boost immunity","Avoid excessive Nitrogen"]},
    "rice healthy":{"name":"Rice Healthy","crop":"Rice","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply Urea at tillering stage","Use NPK 15-15-15 at planting","Apply Zinc foliar spray","Use compost"]},
    "cassava mosaic":{"name":"Cassava Mosaic Disease","crop":"Cassava","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Begomovirus transmitted by whitefly Bemisia tabaci","symptoms":"Mosaic pattern of yellow and green on leaves, severe leaf distortion, stunted plant growth.","treatment":["Remove and destroy all infected plants","Control whitefly population with insecticide","Apply mineral oil spray to reduce virus spread","No chemical cure — prevention is key"],"prevention":["Plant certified virus-free cassava cuttings only","Use mosaic-resistant cassava varieties","Control whitefly with neem-based insecticide","Inspect all planting material carefully before use"],"fertilizer":["Apply balanced NPK to maintain plant strength","Use Potassium to boost immunity","Apply Zinc and Boron foliar spray","Avoid excessive Nitrogen"]},
    "cassava brown streak":{"name":"Cassava Brown Streak Disease","crop":"Cassava","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Ipomovirus: Cassava Brown Streak Virus (CBSV)","symptoms":"Yellow patches on leaves, brown streaks on stems, brown necrotic patches inside tubers making them inedible.","treatment":["Remove and destroy all infected plants immediately","Control whitefly vectors with insecticide","No effective chemical treatment available","Replace with certified resistant varieties"],"prevention":["Use CBSD-resistant varieties","Plant certified disease-free cuttings","Control whitefly population","Avoid moving planting material from infected areas"],"fertilizer":["Apply balanced NPK fertilizer","Use Potassium to boost plant immunity","Apply micronutrient foliar spray","Avoid excessive Nitrogen"]},
    "cassava bacterial blight":{"name":"Cassava Bacterial Blight","crop":"Cassava","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Bacterium: Xanthomonas axonopodis pv. manihotis","symptoms":"Angular water-soaked spots on leaves, wilting of shoots, stem cankers, and gummy exudate on stems.","treatment":["Apply copper-based bactericide","Remove infected parts immediately","Disinfect cutting tools with bleach","Destroy severely infected plants"],"prevention":["Use disease-free planting material","Disinfect tools between plants","Plant resistant varieties","Avoid working in fields when plants are wet"],"fertilizer":["Apply Calcium fertilizer to strengthen cell walls","Use balanced NPK fertilizer","Avoid excessive Nitrogen","Apply Potassium to boost immunity"]},
    "cassava healthy":{"name":"Cassava Healthy","crop":"Cassava","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15 at planting","Use Potassium during tuber formation","Apply Zinc foliar spray","Use compost"]},
    "groundnut early leaf spot":{"name":"Groundnut Early Leaf Spot","crop":"Groundnut","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Cercospora arachidicola","symptoms":"Dark brown circular spots on upper leaf surface with distinct yellow halo around each spot.","treatment":["Spray Chlorothalonil or Mancozeb fungicide","Apply every 14 days from 30 days after planting","Remove and destroy infected leaves","Avoid overhead irrigation"],"prevention":["Use certified disease-free seeds","Rotate crops with non-legume crops","Remove all crop debris after harvest","Plant resistant groundnut varieties"],"fertilizer":["Apply Calcium fertilizer","Use balanced NPK fertilizer","Apply Potassium to boost resistance","Use Gypsum to provide Calcium and Sulfur"]},
    "groundnut late leaf spot":{"name":"Groundnut Late Leaf Spot","crop":"Groundnut","emoji":"🟠","severity":"Moderate-High","sev_emoji":"🟠","action":"Act within 3 days","cause":"Fungus: Cercosporidium personatum","symptoms":"Dark brown to black spots predominantly on lower leaf surface, causing severe defoliation.","treatment":["Apply Tebuconazole or Propiconazole fungicide","Spray every 14 days","Remove and destroy infected material","Improve air circulation"],"prevention":["Rotate crops regularly","Use resistant varieties","Remove crop debris","Avoid dense planting"],"fertilizer":["Apply Calcium and Gypsum","Use balanced NPK fertilizer","Apply Potassium to boost immunity","Avoid excessive Nitrogen"]},
    "groundnut rosette":{"name":"Groundnut Rosette Disease","crop":"Groundnut","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Groundnut Rosette Virus transmitted by aphid Aphis craccivora","symptoms":"Severely stunted plants with small mottled chlorotic leaves arranged in a rosette pattern.","treatment":["Control aphid population with insecticide","Remove and destroy all infected plants","Apply mineral oil spray to reduce virus spread","No chemical cure for the virus"],"prevention":["Plant early to avoid peak aphid season","Use rosette-resistant groundnut varieties","Control aphid with neem-based insecticide","Plant barrier crops around the field"],"fertilizer":["Apply balanced NPK to maintain plant strength","Use Zinc and Boron foliar spray","Apply Potassium to boost immunity","Avoid excessive Nitrogen"]},
    "groundnut rust":{"name":"Groundnut Rust","crop":"Groundnut","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Puccinia arachidis","symptoms":"Orange-brown pustules on lower leaf surfaces, yellowing and premature defoliation of leaves.","treatment":["Apply Mancozeb or Propiconazole fungicide","Spray every 14 days","Remove and destroy infected leaves","Apply at first appearance of pustules"],"prevention":["Plant resistant varieties","Rotate crops","Remove crop debris","Monitor fields regularly"],"fertilizer":["Apply Potassium fertilizer to boost resistance","Use balanced NPK","Apply Calcium and Gypsum","Avoid excessive Nitrogen"]},
    "pepper bacterial spot":{"name":"Pepper Bacterial Spot","crop":"Pepper","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Bacterium: Xanthomonas campestris pv. vesicatoria","symptoms":"Small water-soaked spots on leaves turning dark brown with yellow halo. Spots also appear on fruits.","treatment":["Apply copper-based bactericide every 7 days","Remove and destroy infected plant parts","Avoid working with plants when they are wet","Use streptomycin in severe cases"],"prevention":["Use certified disease-free seeds","Avoid overhead irrigation","Rotate crops every season","Disinfect all tools regularly"],"fertilizer":["Apply Calcium fertilizer to strengthen cell walls","Use balanced NPK","Avoid excessive Nitrogen","Apply Potassium to boost immunity"]},
    "pepper healthy":{"name":"Pepper Healthy","crop":"Pepper","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15 fertilizer","Use Calcium during fruiting","Apply Potassium","Use compost"]},
    "onion purple blotch":{"name":"Onion Purple Blotch","crop":"Onion","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Alternaria porri","symptoms":"Small white lesions with purple centers that enlarge and girdle leaves causing them to collapse.","treatment":["Apply Mancozeb or Iprodione fungicide","Spray every 7-10 days","Remove and destroy infected leaves","Improve air circulation"],"prevention":["Use certified disease-free sets","Avoid overhead irrigation","Rotate crops","Remove crop debris"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK","Apply Calcium foliar spray","Avoid excessive Nitrogen"]},
    "apple scab":{"name":"Apple Scab","crop":"Apple","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Venturia inaequalis","symptoms":"Olive-green to dark brown scab-like lesions on leaves and fruits. Severe cases cause leaf drop.","treatment":["Apply Myclobutanil or Captan fungicide","Spray from bud break stage","Remove and destroy infected leaves","Prune trees to improve air circulation"],"prevention":["Plant scab-resistant apple varieties","Remove fallen leaves in autumn","Apply dormant sprays before bud break","Prune for good airflow through canopy"],"fertilizer":["Apply balanced NPK fertilizer","Use Calcium to strengthen fruit","Apply Boron foliar spray","Avoid excessive Nitrogen"]},
    "apple black rot":{"name":"Apple Black Rot","crop":"Apple","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungus: Botryosphaeria obtusa","symptoms":"Brown circular lesions on leaves with purple margins, black rotting of fruits, and cankers on branches.","treatment":["Apply Captan or Thiophanate-methyl fungicide","Remove and destroy infected fruits and branches","Spray every 7-10 days","Prune out all cankers from branches"],"prevention":["Remove mummified fruits from tree and ground","Prune out infected branches","Maintain tree vigor with proper nutrition","Apply dormant copper spray before bud break"],"fertilizer":["Apply Calcium fertilizer","Use balanced NPK fertilizer","Apply Potassium to boost immunity","Avoid excessive Nitrogen"]},
    "grape black rot":{"name":"Grape Black Rot","crop":"Grape","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungus: Guignardia bidwellii","symptoms":"Brown circular lesions on leaves, shriveled black mummified berries remaining on vine.","treatment":["Apply Myclobutanil or Mancozeb fungicide","Spray from early bud break","Remove and destroy infected berries","Repeat application every 7-14 days"],"prevention":["Remove all mummified berries from vine","Prune for good air circulation","Apply early season protective sprays","Remove all infected plant material"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK","Apply Calcium and Boron foliar spray","Avoid excessive Nitrogen"]},
    "apple cedar rust":{"name":"Apple Cedar Rust","crop":"Apple","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Gymnosporangium juniperi-virginianae","symptoms":"Bright orange-yellow spots on upper leaf surface, small black dots in spot centers, orange tube-like structures on leaf undersides in humid weather.","treatment":["Apply Myclobutanil or Propiconazole fungicide","Spray at pink bud stage through petal fall","Remove nearby juniper/cedar trees if possible","Repeat spray every 10-14 days during infection period"],"prevention":["Plant rust-resistant apple varieties","Remove cedar or juniper trees within 2 miles if feasible","Apply protective fungicide before rainy periods","Monitor trees closely in spring"],"fertilizer":["Apply balanced NPK fertilizer","Use Calcium to strengthen leaf tissue","Apply Potassium to boost disease resistance","Avoid excessive Nitrogen"]},
    "apple healthy":{"name":"Apple Healthy","crop":"Apple","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good orchard sanitation","Prune annually for airflow","Monitor regularly for early signs","Apply preventive dormant sprays"],"fertilizer":["Apply balanced NPK fertilizer in early spring","Use Calcium to strengthen fruit","Apply Boron foliar spray during bloom","Use compost to improve soil health"]},
    "blueberry healthy":{"name":"Blueberry Healthy","crop":"Blueberry","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain acidic soil pH (4.5-5.5)","Monitor regularly for early signs","Ensure good drainage","Prune annually for airflow"],"fertilizer":["Apply acid-forming fertilizer (ammonium sulfate)","Use balanced NPK for acid-loving plants","Apply Sulfur to maintain soil acidity","Use pine bark mulch to retain acidity"]},
    "cherry healthy":{"name":"Cherry Healthy","crop":"Cherry","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good orchard sanitation","Prune for good air circulation","Monitor regularly for early signs","Apply preventive dormant sprays"],"fertilizer":["Apply balanced NPK fertilizer in early spring","Use Calcium to strengthen fruit","Apply Potassium during fruiting stage","Use compost to improve soil health"]},
    "cherry powdery mildew":{"name":"Cherry Powdery Mildew","crop":"Cherry","emoji":"🟡","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Podosphaera clandestina","symptoms":"White powdery patches on leaves, shoots and fruit, leaf curling and distortion, stunted new growth.","treatment":["Apply sulfur-based or Myclobutanil fungicide","Spray at first sign of white patches","Prune infected shoots and destroy them","Repeat treatment every 7-10 days"],"prevention":["Plant resistant cherry varieties","Prune for good air circulation","Avoid excessive nitrogen fertilization","Remove and destroy fallen infected leaves"],"fertilizer":["Apply balanced NPK fertilizer","Avoid excessive Nitrogen which promotes soft growth","Apply Potassium to boost resistance","Use Calcium foliar spray"]},
    "grape black measles":{"name":"Grape Black Measles (Esca)","crop":"Grape","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungal complex including Phaeomoniella chlamydospora","symptoms":"Tiger-stripe pattern of yellow and brown discoloration between leaf veins, dark spots on berries, sudden vine collapse (apoplexy) in severe cases.","treatment":["Prune out and destroy infected wood well below symptoms","Apply wound-protectant paste after pruning","Avoid pruning during wet weather","No curative fungicide exists — focus on removal"],"prevention":["Prune during dry weather only","Disinfect pruning tools between vines","Avoid large pruning wounds","Remove and burn severely infected vines"],"fertilizer":["Apply balanced NPK fertilizer","Avoid water stress and over-irrigation","Apply Potassium to support vine vigor","Maintain overall plant health to reduce susceptibility"]},
    "grape healthy":{"name":"Grape Healthy","crop":"Grape","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good vineyard sanitation","Prune for good air circulation","Monitor regularly for early signs","Remove fallen leaves and debris"],"fertilizer":["Apply balanced NPK fertilizer in spring","Use Potassium during fruit development","Apply Calcium and Boron foliar spray","Use compost to improve soil health"]},
    "grape leaf blight":{"name":"Grape Leaf Blight (Isariopsis)","crop":"Grape","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Pseudocercospora vitis (Isariopsis leaf spot)","symptoms":"Angular brown to reddish-brown spots on leaves bounded by veins, spots may merge causing premature leaf drop.","treatment":["Apply Mancozeb or copper-based fungicide","Spray every 10-14 days during wet season","Remove and destroy infected leaves","Improve canopy airflow through pruning"],"prevention":["Prune for good air circulation","Avoid overhead irrigation","Remove fallen infected leaves after harvest","Apply preventive fungicide before rainy season"],"fertilizer":["Apply balanced NPK fertilizer","Use Potassium to boost resistance","Apply Calcium foliar spray","Avoid excessive Nitrogen"]},
    "orange citrus greening":{"name":"Orange Citrus Greening (HLB)","crop":"Orange","emoji":"🔴","severity":"Severe","sev_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Bacterium: Candidatus Liberibacter, spread by Asian citrus psyllid","symptoms":"Blotchy yellow mottling on leaves (asymmetric across midrib), small lopsided bitter fruit, premature fruit drop, twig dieback.","treatment":["Remove and destroy infected trees to prevent spread","Control Asian citrus psyllid with systemic insecticide","No cure exists once a tree is infected","Apply nutritional foliar sprays to prolong tree productivity"],"prevention":["Plant certified disease-free nursery stock","Control psyllid populations aggressively","Inspect trees regularly for early symptoms","Avoid moving citrus plant material from infected areas"],"fertilizer":["Apply balanced NPK with micronutrients","Use Zinc, Manganese and Boron foliar spray","Apply Magnesium to correct common deficiency","Maintain consistent irrigation to reduce stress"]},
    "peach bacterial spot":{"name":"Peach Bacterial Spot","crop":"Peach","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 3 days","cause":"Bacterium: Xanthomonas arboricola pv. pruni","symptoms":"Small dark angular spots on leaves that may fall out leaving a shot-hole appearance, dark sunken lesions on fruit.","treatment":["Apply copper-based bactericide before bud break","Avoid copper sprays during hot weather (can burn leaves)","Remove and destroy severely infected shoots","Repeat copper application in early spring"],"prevention":["Plant resistant peach varieties","Avoid overhead irrigation","Prune for good air circulation","Avoid excessive nitrogen fertilization"],"fertilizer":["Apply balanced NPK fertilizer","Avoid excessive Nitrogen","Apply Calcium to strengthen cell walls","Use Potassium to boost immunity"]},
    "peach healthy":{"name":"Peach Healthy","crop":"Peach","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good orchard sanitation","Prune annually for airflow","Monitor regularly for early signs","Apply preventive dormant sprays"],"fertilizer":["Apply balanced NPK fertilizer in early spring","Use Calcium to strengthen fruit","Apply Potassium during fruiting stage","Use compost to improve soil health"]},
    "raspberry healthy":{"name":"Raspberry Healthy","crop":"Raspberry","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good drainage","Prune old canes after fruiting","Monitor regularly for early signs","Ensure good air circulation between canes"],"fertilizer":["Apply balanced NPK fertilizer in early spring","Use Potassium during fruiting stage","Apply Calcium and Magnesium foliar spray","Use compost to improve soil health"]},
    "soybean healthy":{"name":"Soybean Healthy","crop":"Soybean","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Practice crop rotation","Use certified disease-free seeds","Monitor regularly for early signs","Maintain proper field drainage"],"fertilizer":["Apply Phosphorus and Potassium at planting","Use Rhizobium inoculant for nitrogen fixation","Apply Sulfur if soil is deficient","Use compost to improve soil health"]},
    "squash powdery mildew":{"name":"Squash Powdery Mildew","crop":"Squash","emoji":"🟡","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Podosphaera xanthii","symptoms":"White powdery spots on upper and lower leaf surfaces, leaves yellow and shrivel, reduced fruit yield and quality.","treatment":["Apply sulfur or potassium bicarbonate spray","Spray at first sign of white patches","Remove and destroy heavily infected leaves","Repeat treatment every 7 days"],"prevention":["Plant resistant squash varieties","Ensure good air circulation between plants","Avoid overhead watering","Avoid excessive nitrogen fertilization"],"fertilizer":["Apply balanced NPK fertilizer","Avoid excessive Nitrogen which promotes soft growth","Apply Potassium to boost resistance","Use Silicon fertilizer to strengthen leaf tissue"]},
    "strawberry healthy":{"name":"Strawberry Healthy","crop":"Strawberry","emoji":"🟢","severity":"Healthy","sev_emoji":"🟢","action":"No action needed","cause":"No disease detected","symptoms":"Plant appears completely healthy with no visible disease symptoms.","treatment":["No treatment needed","Continue regular care and monitoring"],"prevention":["Maintain good drainage","Use mulch to prevent soil splash","Monitor regularly for early signs","Space plants properly for airflow"],"fertilizer":["Apply balanced NPK fertilizer","Use Calcium to strengthen fruit","Apply Potassium during fruiting stage","Use compost to improve soil health"]},
    "strawberry leaf scorch":{"name":"Strawberry Leaf Scorch","crop":"Strawberry","emoji":"🟠","severity":"Moderate","sev_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Diplocarpon earlianum","symptoms":"Small dark purple spots on leaves that enlarge and merge, giving a scorched or burned appearance, reduced plant vigor.","treatment":["Apply Captan or copper-based fungicide","Remove and destroy infected leaves","Spray every 10-14 days during wet season","Improve air circulation around plants"],"prevention":["Plant resistant strawberry varieties","Space plants properly for airflow","Avoid overhead irrigation","Remove old infected leaves after harvest"],"fertilizer":["Apply balanced NPK fertilizer","Use Potassium to boost resistance","Apply Calcium foliar spray","Avoid excessive Nitrogen"]},
}

# ── Crop helpers (used by the crop check) ──────
def class_crop(name):
    """Crop that a model class belongs to, e.g. 'Corn Common Rust' -> 'Corn/Maize'."""
    first = name.split()[0]
    return "Corn/Maize" if first == "Corn" else first

# crops the photo model was trained on, and crops that only work through Search/Browse
photo_crops = sorted({class_crop(c) for c in ai_classes})
search_only_crops = sorted({i["crop"] for i in diseases_db.values()} - set(photo_crops))
NO_CROP = "— Select your crop —"

# ── CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, * { font-family: 'Inter', sans-serif !important; }

/* Exclude Streamlit's icon fonts from the override above so icon
   glyphs render as icons, not literal overlapping text. */
[data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzoneIcon"],
[data-testid="stExpanderIcon"],
span[class*="material-icons"],
i[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
}

.main { background: #f0f4f0 !important; }
.block-container { padding: 1.5rem 1rem !important; max-width: 820px !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
    border-radius: 20px; padding: 48px 32px; text-align: center;
    margin-bottom: 20px; box-shadow: 0 8px 32px rgba(27,94,32,0.3);
}
.hero-tag {
    display: inline-block; background: rgba(255,255,255,0.15);
    color: white; border: 1px solid rgba(255,255,255,0.3);
    padding: 5px 16px; border-radius: 100px; font-size: 11px;
    font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 16px;
}
.hero h1 { color: white; font-size: 1.5rem; font-weight: 700; margin: 0 0 12px 0; line-height: 1.35; letter-spacing: 0.5px; }
.hero p { color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0; }

.stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.stat { background: white; border-radius: 14px; padding: 18px 10px; text-align: center; border: 1px solid #e0e8e0; }
.stat-n { font-size: 1.8rem; font-weight: 700; color: #1b5e20; line-height: 1; margin-bottom: 4px; }
.stat-l { font-size: 10px; color: #90a090; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] { background: white; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #e0e8e0; margin-bottom: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #6c757d; font-weight: 500; font-size: 0.85rem; background: transparent; }
.stTabs [aria-selected="true"] { background: #1b5e20 !important; color: white !important; font-weight: 600 !important; }

.stButton > button { background: #1b5e20 !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: 600 !important; font-size: 1rem !important; padding: 14px !important; }
.stButton > button:hover { background: #2e7d32 !important; transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(27,94,32,0.35) !important; }

.card { background: white; border-radius: 14px; padding: 22px 24px; margin: 12px 0; border: 1px solid #e0e8e0; }
.card-disease { border-left: 4px solid #c62828; }
.card-healthy { border-left: 4px solid #1b5e20; }
.card-warning { border-left: 4px solid #f57c00; }

.tag { display: inline-block; padding: 4px 12px; border-radius: 100px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px; }
.tag-red { background: #ffebee; color: #c62828; }
.tag-green { background: #e8f5e9; color: #1b5e20; }
.tag-orange { background: #fff3e0; color: #e65100; }

.card-title { font-size: 1.3rem; font-weight: 700; color: #1a1a1a; margin: 0 0 6px 0; }
.card-sub { color: #6c757d; font-size: 0.85rem; line-height: 1.5; }

.conf-box { background: white; border-radius: 14px; padding: 24px; text-align: center; margin: 12px 0; border: 1px solid #e0e8e0; }
.conf-lbl { font-size: 10px; font-weight: 700; color: #90a090; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
.conf-val { font-size: 4rem; font-weight: 700; color: #1b5e20; line-height: 1; }
.conf-sub { color: #90a090; font-size: 0.8rem; margin-top: 6px; }

.sev-box { background: white; border: 1px solid #ffe0b2; border-radius: 14px; padding: 18px 22px; margin: 12px 0; display: flex; align-items: center; gap: 14px; }
.sev-icon { font-size: 2rem; flex-shrink: 0; }
.sev-title { font-size: 10px; font-weight: 700; color: #e65100; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.sev-level { font-size: 1.1rem; font-weight: 700; color: #1a1a1a; margin-bottom: 2px; }
.sev-action { color: #6c757d; font-size: 0.85rem; }

.info-box { background: white; border-radius: 14px; padding: 20px 24px; margin: 10px 0; border: 1px solid #e0e8e0; }
.info-title { font-size: 11px; font-weight: 700; color: #495057; text-transform: uppercase; letter-spacing: 1px; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #f0f4f0; }
.info-row { margin-bottom: 10px; }
.info-row:last-child { margin-bottom: 0; }
.info-lbl { font-size: 10px; font-weight: 700; color: #90a090; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.info-val { font-size: 0.88rem; color: #343a40; line-height: 1.5; }

.li { display: flex; gap: 10px; padding: 9px 0; border-bottom: 1px solid #f8f9fa; font-size: 0.87rem; color: #495057; line-height: 1.5; }
.li:last-child { border-bottom: none; }
.dg { color: #1b5e20; font-weight: 700; flex-shrink: 0; }
.db { color: #1565c0; font-weight: 700; flex-shrink: 0; }
.dp { color: #6a1b9a; font-weight: 700; flex-shrink: 0; }

.chip { display: inline-block; background: #f0f4f0; border: 1px solid #e0e8e0; border-radius: 100px; padding: 5px 12px; font-size: 12px; color: #495057; margin: 3px; font-weight: 500; }
.badge { display: inline-flex; align-items: center; gap: 5px; background: #e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9; border-radius: 100px; padding: 5px 12px; font-size: 12px; font-weight: 600; margin: 3px; }

.how-box { background: white; border-radius: 14px; padding: 20px 24px; margin: 16px 0; border: 1px solid #e0e8e0; }
.how-title { font-size: 11px; font-weight: 700; color: #495057; text-transform: uppercase; letter-spacing: 1px; padding-bottom: 12px; margin-bottom: 4px; border-bottom: 1px solid #f0f4f0; }
.step { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f8f9fa; }
.step:last-child { border-bottom: none; }
.step-n { width: 26px; height: 26px; background: #e8f5e9; color: #1b5e20; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.step-t { font-size: 0.87rem; color: #495057; line-height: 1.5; padding-top: 3px; }

.crop-head { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 1px solid #c8e6c9; border-radius: 14px; padding: 16px 20px; margin: 12px 0; display: flex; align-items: center; gap: 14px; }
.crop-emoji { font-size: 2.2rem; flex-shrink: 0; }
.crop-name { font-size: 1.2rem; font-weight: 700; color: #1b5e20; margin-bottom: 2px; }
.crop-sub { font-size: 0.8rem; color: #388e3c; }

.footer { background: white; border: 1px solid #e0e8e0; border-radius: 14px; padding: 24px; text-align: center; margin-top: 40px; }
.footer-logo { font-size: 1.2rem; font-weight: 700; color: #1b5e20; margin-bottom: 8px; }
.footer-text { font-size: 12px; color: #90a090; line-height: 2; }
.footer-link { color: #1b5e20; font-weight: 600; text-decoration: none; }

.stTextInput > div > div > input { background: white !important; border: 1px solid #e0e8e0 !important; border-radius: 10px !important; font-size: 0.9rem !important; padding: 12px 14px !important; color: #343a40 !important; }
.stTextInput > div > div > input:focus { border-color: #1b5e20 !important; box-shadow: 0 0 0 3px rgba(27,94,32,0.1) !important; }
.stSelectbox > div > div { background: white !important; border: 1px solid #e0e8e0 !important; border-radius: 10px !important; }
.stRadio > div { background: white; border-radius: 12px; padding: 12px 16px; border: 1px solid #e0e8e0; }

[data-testid="stFileUploaderDropzone"] {
    background: #f7faf7 !important;
    border: 2px dashed #a5d6a7 !important;
    border-radius: 14px !important;
    padding: 24px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: white !important;
    color: #1b5e20 !important;
    border: 1.5px solid #1b5e20 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #1b5e20 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helper Function ────────────────────────────
def list_box(title, items, cls):
    # FIXED: build the whole card in ONE block so the bullet items stay inside the card
    rows = "".join(f'<div class="li"><span class="{cls}">▸</span><span>{t}</span></div>' for t in items)
    st.markdown(f'<div class="info-box"><div class="info-title">{title}</div>{rows}</div>', unsafe_allow_html=True)

def show_info(info, confidence=None):
    if confidence is not None:
        st.markdown(f"""
        <div class="conf-box">
            <div class="conf-lbl">Confidence Score</div>
            <div class="conf-val">{confidence:.1f}%</div>
            <div class="conf-sub">Based on MobileNetV2 image analysis</div>
        </div>
        """, unsafe_allow_html=True)

    if info['severity'] != 'Healthy':
        st.markdown(f"""
        <div class="sev-box">
            <div class="sev-icon">{info['sev_emoji']}</div>
            <div>
                <div class="sev-title">Disease Severity Level</div>
                <div class="sev-level">{info['severity']}</div>
                <div class="sev-action">⏰ {info['action']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <div class="info-title">🔬 Disease Information</div>
        <div class="info-row">
            <div class="info-lbl">Disease Name</div>
            <div class="info-val">{info['name']}</div>
        </div>
        <div class="info-row">
            <div class="info-lbl">Affected Crop</div>
            <div class="info-val">{info['crop']}</div>
        </div>
        <div class="info-row">
            <div class="info-lbl">Cause</div>
            <div class="info-val">{info['cause']}</div>
        </div>
        <div class="info-row">
            <div class="info-lbl">Symptoms</div>
            <div class="info-val">{info['symptoms']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    list_box("💊 Recommended Treatment", info["treatment"], "dg")

    list_box("🛡️ Prevention Tips", info["prevention"], "db")

    list_box("🌱 Fertilizer Recommendation", info["fertilizer"], "dp")

# ── Hero ───────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🌿 Agriculture Support — FUTB 2024/2025</div>
    <h1>FARMER SUPPORT DECISION SYSTEM<br>FOR PLANT DISEASE DETECTION<br>AND TREATMENT</h1>
    <p>Upload or take a leaf photo · Get an instant diagnosis · Follow treatment advice</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats">
    <div class="stat"><div class="stat-n">18</div><div class="stat-l">Crops</div></div>
    <div class="stat"><div class="stat-n">39</div><div class="stat-l">Diseases</div></div>
    <div class="stat"><div class="stat-n">95%</div><div class="stat-l">Accuracy</div></div>
    <div class="stat"><div class="stat-n">Free</div><div class="stat-l">Always</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📸 Detect Disease",
    "🔍 Search",
    "🌾 Browse Crops"
])

# ═══════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="how-box">
        <div class="how-title">📖 How to Use</div>
        <div class="step">
            <div class="step-n">1</div>
            <div class="step-t">Select the crop your leaf comes from</div>
        </div>
        <div class="step">
            <div class="step-n">2</div>
            <div class="step-t">Upload a photo from gallery OR take a photo directly with your camera</div>
        </div>
        <div class="step">
            <div class="step-n">3</div>
            <div class="step-t">Click Analyze — the system checks if it is a leaf, then detects the disease</div>
        </div>
        <div class="step">
            <div class="step-n">4</div>
            <div class="step-t">Read the diagnosis, severity level, treatment and fertilizer advice</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ✅ Crops Supported by Photo Detection")
    st.markdown("""
    <div style="margin:8px 0 4px 0;">
        <span class="chip">🍎 Apple</span>
        <span class="chip">🫐 Blueberry</span>
        <span class="chip">🍒 Cherry</span>
        <span class="chip">🌽 Corn/Maize</span>
        <span class="chip">🍇 Grape</span>
        <span class="chip">🍊 Orange</span>
        <span class="chip">🍑 Peach</span>
        <span class="chip">🌶️ Pepper</span>
        <span class="chip">🥔 Potato</span>
        <span class="chip">🍓 Raspberry</span>
        <span class="chip">🫘 Soybean</span>
        <span class="chip">🎃 Squash</span>
        <span class="chip">🍓 Strawberry</span>
        <span class="chip">🍅 Tomato</span>
    </div>
    <p style="color:#90a090;font-size:0.78rem;margin:6px 0 16px 0;">
    ⚠️ Rice · Cassava · Groundnut · Onion — Use Search or Browse tab
    </p>
    """, unsafe_allow_html=True)

    # ── Step 1: crop selection ─────────────────
    st.markdown("#### 🌱 Step 1: Select Your Crop")
    crop_choice = st.selectbox(
        "Which crop is this leaf from?",
        [NO_CROP] + photo_crops + search_only_crops + ["Not sure"],
        label_visibility="collapsed"
    )

    # ── Step 2: camera or upload ───────────────
    st.markdown("#### 📷 Step 2: Choose How to Add Your Leaf Photo")
    method = st.radio(
        "Input method:",
        ["📁 Upload from Gallery", "📷 Take Photo with Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )

    image = None
    if method == "📁 Upload from Gallery":
        uploaded = st.file_uploader(
            "Upload plant leaf image",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        camera = st.camera_input("Take a photo of the plant leaf")
        if camera:
            image = Image.open(camera).convert("RGB")

    if image:
        c1, c2, c3 = st.columns([1,4,1])
        with c2:
            st.image(
                image,
                caption="✅ Your leaf photo is ready for analysis",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button(
            "⚡  ANALYZE DISEASE NOW",
            use_container_width=True,
            type="primary"
        )

        if btn:
            if crop_choice == NO_CROP:
                st.warning("⚠️ Please select your crop in Step 1, then click Analyze.")
            elif crop_choice in search_only_crops:
                # The photo model was not trained on these crops, so do not guess from the photo.
                st.markdown(f"""
                <div class="card card-warning">
                    <div class="tag tag-orange">ℹ️ Photo Detection Not Available</div>
                    <div class="card-title">{crop_choice}: Use the Disease Guide</div>
                    <div class="card-sub">
                    The photo model was not trained on {crop_choice} leaves, so it cannot
                    diagnose them from a photo. Compare your leaf with the diseases listed
                    below and check the symptoms and treatment of each one.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                for info in [m for m in diseases_db.values() if m["crop"] == crop_choice]:
                    with st.expander(f"{info['emoji']} {info['name']} — {info['severity']}"):
                        show_info(info)
            elif not model_loaded:
                st.error("❌ Model not loaded! Make sure your .tflite files are in the models/ folder.")
            else:
                img = image.resize((224,224))
                arr = np.array(img, dtype=np.float32)/255.0
                inp = np.expand_dims(arr, axis=0)

                # Stage 1 — Leaf Validation
                if validator_loaded:
                    with st.spinner("🔍 Stage 1 — Checking if this is a plant leaf..."):
                        validator.set_tensor(val_input[0]['index'], inp)
                        validator.invoke()
                        vp = validator.get_tensor(val_output[0]['index'])
                        is_not_leaf = np.argmax(vp) == 1
                        v_conf = np.max(vp)*100

                    if is_not_leaf and v_conf > 70:
                        st.markdown("""
                        <div class="card card-warning">
                            <div class="tag tag-orange">❌ Invalid Image</div>
                            <div class="card-title">This is NOT a Plant Leaf!</div>
                            <div class="card-sub">
                            Our Computer Vision system detected that the uploaded
                            image is not a plant leaf. Please upload a proper
                            plant leaf photo and try again.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="how-box">
                            <div class="how-title">💡 Tips for a Good Photo</div>
                            <div class="step">
                                <div class="step-n">✓</div>
                                <div class="step-t">Take a close-up photo of the leaf only</div>
                            </div>
                            <div class="step">
                                <div class="step-n">✓</div>
                                <div class="step-t">Leaf should fill most of the photo frame</div>
                            </div>
                            <div class="step">
                                <div class="step-n">✓</div>
                                <div class="step-t">Use good natural daylight</div>
                            </div>
                            <div class="step">
                                <div class="step-n" style="background:#ffebee;color:#c62828;">✗</div>
                                <div class="step-t" style="color:#c62828;">
                                Do NOT upload photos of people, animals, cars or objects
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()

                # Stage 2 — Disease Detection
                with st.spinner("🔬 Stage 2 — Analyzing disease..."):
                    interpreter.set_tensor(input_details[0]['index'], inp)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    result = ai_classes[np.argmax(pred)]
                    conf = np.max(pred)*100

                # Crop check: the predicted crop must match the crop the user selected
                crop_mismatch = (crop_choice != "Not sure" and class_crop(result) != crop_choice)

                st.markdown("<br>", unsafe_allow_html=True)

                if validator_loaded:
                    st.markdown("""
                    <div style="margin-bottom:16px;">
                        <span class="badge">✅ Stage 1: Leaf Confirmed</span>
                        <span class="badge">✅ Stage 2: Analysis Complete</span>
                    </div>
                    """, unsafe_allow_html=True)

                if crop_choice == "Not sure" and conf >= 60 and not crop_mismatch:
                    st.warning("⚠️ You did not confirm the crop, so this result may be wrong for crops the model was not trained on (Rice, Cassava, Groundnut, Onion). Please check that the detected crop matches your plant.")

                if conf < 60 or crop_mismatch:
                    # A low score, or a predicted crop that differs from the selected crop,
                    # means the leaf is probably not in the training data.
                    if crop_mismatch:
                        low_msg = (f"This leaf did not match the {crop_choice} diseases "
                                   f"the model knows (score: {conf:.1f}%). Check that you "
                                   f"selected the correct crop, or use the Search or Browse "
                                   f"tab to find its diseases and treatments.")
                    else:
                        low_msg = (f"The system could not identify this leaf with enough "
                                   f"confidence (score: {conf:.1f}%, required: 60%). This crop "
                                   f"or disease may not be in the model's training data. "
                                   f"If your crop is Rice, Cassava, Groundnut or Onion, use "
                                   f"the Search or Browse tab to get its diseases and treatments.")
                    st.markdown(f"""
                    <div class="card card-warning">
                        <div class="tag tag-orange">⚠️ Leaf Not Recognized</div>
                        <div class="card-title">Leaf Not Recognized</div>
                        <div class="card-sub">{low_msg}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="how-box">
                        <div class="how-title">💡 What You Can Do</div>
                        <div class="step">
                            <div class="step-n">1</div>
                            <div class="step-t"><b>Check the crop</b> — Photo detection only works for the supported crops listed above</div>
                        </div>
                        <div class="step">
                            <div class="step-n">2</div>
                            <div class="step-t"><b>Use Search or Browse</b> — For Rice, Cassava, Groundnut, Onion or any crop not listed, open the Search or Browse Crops tab</div>
                        </div>
                        <div class="step">
                            <div class="step-n">3</div>
                            <div class="step-t"><b>Search by symptom</b> — In the Search tab, type what you see on the leaf, for example "yellow spots"</div>
                        </div>
                        <div class="step">
                            <div class="step-n">4</div>
                            <div class="step-t"><b>Try another photo</b> — If your crop is supported, take a close-up of one leaf in natural light</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                elif "Healthy" in result:
                    st.markdown(f"""
                    <div class="card card-healthy">
                        <div class="tag tag-green">✅ Healthy Plant</div>
                        <div class="card-title">{result}</div>
                        <div class="card-sub">
                        Great news! Your plant appears completely healthy.
                        No disease was detected. Keep monitoring regularly.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    info = diseases_db.get(result.lower())
                    if info:
                        show_info(info, conf)

                else:
                    st.markdown(f"""
                    <div class="card card-disease">
                        <div class="tag tag-red">⚠️ Disease Detected</div>
                        <div class="card-title">{result}</div>
                        <div class="card-sub">
                        Disease identified! Please follow the treatment
                        recommendations below immediately to prevent
                        further spread across your farm.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    info = diseases_db.get(result.lower())
                    if info:
                        show_info(info, conf)
                    else:
                        st.markdown(f"""
                        <div class="conf-box">
                            <div class="conf-lbl">Confidence Score</div>
                            <div class="conf-val">{conf:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("Use the Search tab to find more information about this disease.")

# ═══════════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Search Any Disease or Crop")
    st.markdown("""
    <p style="color:#90a090;margin-bottom:16px;font-size:0.9rem;">
    Type any crop name, disease name or symptom to get complete information
    </p>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "",
        placeholder="🔎  Try: tomato, rice, cassava, blight, rust, maize, groundnut...",
        label_visibility="collapsed"
    )

    if not query:
        st.markdown("""
        <div style="margin-top:8px;">
            <span class="chip">🍅 Tomato</span>
            <span class="chip">🌾 Rice</span>
            <span class="chip">🌿 Cassava</span>
            <span class="chip">🌽 Maize</span>
            <span class="chip">🥜 Groundnut</span>
            <span class="chip">🥔 Potato</span>
            <span class="chip">🌶️ Pepper</span>
            <span class="chip">🍎 Apple</span>
            <span class="chip">🍇 Grape</span>
        </div>
        """, unsafe_allow_html=True)

    if query:
        q = query.lower().strip()
        found = [
            info for key, info in diseases_db.items()
            if q in key
            or q in info["name"].lower()
            or q in info["crop"].lower()
            or q in info["symptoms"].lower()
            or q in info["cause"].lower()
        ]
        if found:
            st.success(f"✅ Found **{len(found)}** result(s) for **'{query}'**")
            for info in found:
                label = f"{info['emoji']} {info['name']} | {info['crop']} | {info['severity']}"
                with st.expander(label):
                    show_info(info)
        else:
            st.markdown(f"""
            <div class="card card-warning">
                <div class="tag tag-orange">No Results</div>
                <div class="card-title">"{query}" not found</div>
                <div class="card-sub">
                Try: tomato, potato, rice, cassava, maize,
                groundnut, pepper, onion, apple, grape, blight, rust
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌾 Browse All Diseases by Crop")
    st.markdown("""
    <p style="color:#90a090;margin-bottom:16px;font-size:0.9rem;">
    Select any crop to see all its diseases, treatments and fertilizer advice
    </p>
    """, unsafe_allow_html=True)

    crops = sorted(set(i["crop"] for i in diseases_db.values()))
    crop_emojis = {
        "Apple":"🍎","Blueberry":"🫐","Cassava":"🌿","Cherry":"🍒","Corn/Maize":"🌽",
        "Grape":"🍇","Groundnut":"🥜","Onion":"🧅","Orange":"🍊","Peach":"🍑",
        "Pepper":"🌶️","Potato":"🥔","Raspberry":"🍓","Rice":"🌾","Soybean":"🫘",
        "Squash":"🎃","Strawberry":"🍓","Tomato":"🍅"
    }

    selected = st.selectbox(
        "",
        ["— Select a crop to view diseases —"] + crops,
        label_visibility="collapsed"
    )

    if selected != "— Select a crop to view diseases —":
        matches = [i for i in diseases_db.values() if i["crop"] == selected]
        emoji = crop_emojis.get(selected, "🌿")
        dc = len([m for m in matches if "Healthy" not in m["name"]])

        st.markdown(f"""
        <div class="crop-head">
            <div class="crop-emoji">{emoji}</div>
            <div>
                <div class="crop-name">{selected}</div>
                <div class="crop-sub">
                {dc} disease(s) found · Full treatment guide included
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        diseases_list = [m for m in matches if "Healthy" not in m["name"]]
        healthy_list  = [m for m in matches if "Healthy" in m["name"]]

        if diseases_list:
            st.markdown(f"#### ⚠️ {len(diseases_list)} Disease(s)")
            for info in diseases_list:
                label = f"{info['name']} — {info['severity']}"
                with st.expander(label):
                    show_info(info)

        if healthy_list:
            st.markdown("#### ✅ Healthy State")
            for info in healthy_list:
                with st.expander(info['name']):
                    show_info(info)

# ── Footer ─────────────────────────────────────
# CHANGED: removed the stray blank lines and the leftover word "System"
st.markdown("""
<div class="footer">
    <div class="footer-logo">🌿 farmersupport </div>
    <div class="footer-text">FARMER SUPPORT DECISION SYSTEM FOR PLANT DISEASE DETECTION AND TREATMEN
        T<br>
        Developed by <b style="color:#343a40;">Yusuf Gambo</b>
        &nbsp;·&nbsp; Matric No: SIT/CSC/23/0005<br>
        B.Sc Computer Science &nbsp;·&nbsp;
        Federal University of Technology Babura &nbsp;·&nbsp; 2024/2025<br>
        Supervised by <b style="color:#343a40;">Dr. Khalid Haruna</b>
        <br><br>
    </div>
</div>
""", unsafe_allow_html=True)
