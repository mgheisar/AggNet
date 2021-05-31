# notice that VGGFace2
# contains 508 overlapping identities with LFW and 205 with YTF datasets, therefore, we
# removed overlapping identities from both datasets to report fair results.
import os

# with open('lfw_identity_meta.txt', 'r') as lfw_file:
#     lines = lfw_file.readlines()
#     lfw_identity_list = [str(line.split('\t')[0].lower().strip()) for line in lines]
#     lfw_identity_list = sorted(set(lfw_identity_list))
#
# # print(len(lfw_identity_list))
# # print(lfw_identity_list)
#
# with open('vggface2_identity_meta.csv', 'r') as vggface2_file:
#     lines = vggface2_file.readlines()
#     vggface2_identity_list = [str(line.split(',')[1].lower().replace('\"', '').strip()) for line in lines]
#     vggface2_identity_list = sorted(set(vggface2_identity_list))
#
# # print(len(vggface2_identity_list))
# # print(vggface2_identity_list)
#
# overlap_list = [identity for identity in lfw_identity_list if identity in vggface2_identity_list]
# print(len(overlap_list))
# print(overlap_list)

a = ['aaron_tippin', 'ai_sugiyama', 'al_sharpton', 'alan_mulally', 'albert_pujols', 'alberto_fujimori', 'alek_wek',
     'alessandro_nesta', 'alex_ferguson', 'alexa_vega', 'alexander_payne', 'alexis_bledel', 'alfonso_portillo',
     'allyson_felix', 'alyson_hannigan', 'amanda_beard', 'amelia_vega', 'amy_brenneman', 'amy_pascal', 'amy_smart',
     'andrea_bocelli', 'andy_lau', 'andy_roddick', 'angela_bassett', 'anne_heche', 'annette_bening', 'anthony_hopkins',
     'antonio_banderas', 'antonio_cassano', 'arianna_huffington', 'aron_ralston', 'ashley_olsen', 'ashraf_ghani',
     'ashton_kutcher', 'avril_lavigne', 'barbara_bach', 'barbara_boxer', 'barbra_streisand', 'barry_bonds',
     'bart_freundlich', 'ben_howland', 'ben_lee', 'bill_belichick', 'bill_parcells', 'bill_self', 'bill_walton',
     'billy_crystal', 'billy_donovan', 'blythe_danner', 'bo_pelini', 'bo_ryan', 'bob_huggins', 'bob_iger', 'bob_melvin',
     'bob_menendez', 'bob_stoops', 'boris_becker', 'boris_berezovsky', 'brad_garrett', 'brad_gushue', 'brad_pitt',
     'brandon_knight', 'brian_cowen', 'britney_spears', 'brittany_snow', 'bronson_arroyo', 'bruce_willis',
     'byron_scott', 'calista_flockhart', 'camryn_manheim', 'carla_gugino', 'carlo_ancelotti', 'carlos_queiroz',
     'carly_fiorina', 'caroline_dhavernas', 'carrie-anne_moss', 'carson_daly', 'cecilia_bolocco', 'cedric_benson',
     'charlie_hunnam', 'charlie_sheen', 'charlotte_casiraghi', 'charlotte_church', 'charlotte_rampling', 'cheryl_hines',
     'cheryl_tiegs', 'chris_dodd', 'chris_pronger', 'christian_wulff', 'christine_ebersole', 'christoph_daum',
     'cindy_margolis', 'ciro_gomes', 'claire_danes', 'claudia_schiffer', 'colin_farrell', 'colin_montgomerie',
     'conan_obrien', 'corinna_harfouch', 'craig_david', 'crispin_glover', 'cristina_saralegui', 'cynthia_nixon',
     'daisy_fuentes', 'dale_earnhardt', 'daniel_zelman', 'danny_ainge', 'danny_elfman', 'danny_green',
     'dario_franchitti', 'david_arquette', 'david_beckham', 'david_bisbal', 'david_caruso', 'david_coulthard',
     'david_hasselhoff', 'david_spade', 'david_stern', 'davis_love_iii', 'dean_sheremet', 'debra_messing', 'demi_moore',
     'denise_van_outen', 'deniz_baykal', 'denzel_washington', 'diana_krall', 'diane_lane', 'dionne_warwick',
     'doc_rivers', 'dominic_monaghan', 'dominique_de_villepin', 'donald_trump', 'donatella_versace', 'donna_brazile',
     'donny_osmond', 'drew_barrymore', 'dustin_hoffman', 'dwayne_johnson', 'ed_rendell', 'edgar_savisaar', 'edie_falco',
     'edward_burns', 'edward_norton', 'el_hadji_diouf', 'elisha_cuthbert', 'eliza_dushku', 'elizabeth_hurley',
     'elizabeth_smart', 'ellen_barkin', 'ellen_pompeo', 'elvis_costello', 'elvis_presley', 'emma_thompson',
     'emmy_rossum', 'eric_bana', 'eric_christian_olsen', 'erika_christensen', 'estella_warren', 'eva_mendes',
     'eve_ensler', 'evgeni_plushenko', 'ewan_mcgregor', 'fann_wong', 'felicity_huffman', 'fernando_alonso',
     'flavia_pennetta', 'frances_fisher', 'francesco_totti', 'frank_beamer', 'gabrielle_union', 'gary_bettman',
     'gary_sinise', 'geoffrey_rush', 'george_gregan', 'george_pataki', 'geraldo_rivera', 'gina_gershon', 'gina_torres',
     'gloria_gaynor', 'graeme_smith', 'greg_kinnear', 'gregg_popovich', 'gregor_gysi', 'gretchen_mol',
     'guillaume_depardieu', 'guus_hiddink', 'guy_verhofstadt', 'halle_berry', 'hank_azaria', 'hartmut_mehdorn',
     'harvey_fierstein', 'harvey_weinstein', 'hassan_nasrallah', 'heather_locklear', 'heather_mills', 'heidi_fleiss',
     'heidi_klum', 'henri_proglio', 'henrique_meirelles', 'holly_hunter', 'holly_robinson_peete', 'ian_mckellen',
     'isabella_rossellini', 'isaiah_washington', 'islam_karimov', 'ivana_trump', 'jack_osbourne', 'jack_straw',
     'jackie_chan', 'jacques_rogge', 'jada_pinkett_smith', 'jaime_pressly', 'jake_gyllenhaal', 'james_cameron',
     'james_franco', 'james_murdoch', 'jamie_lee_curtis', 'jan_peter_balkenende', 'jan_ullrich', 'jane_kaczmarek',
     'jane_krakowski', 'janet_napolitano', 'jason_kidd', 'javier_bardem', 'javier_saviola', 'javier_zanetti',
     'jay_leno', 'jc_chasez', 'jean-claude_juncker', 'jean-pierre_raffarin', 'jean_charest', 'jean_todt',
     'jeff_bridges', 'jeff_hornacek', 'jenna_elfman', 'jennie_finch', 'jennie_garth', 'jennifer_capriati',
     'jennifer_love_hewitt', 'jennifer_tilly', 'jerry_bruckheimer', 'jerry_sloan', 'jesse_jackson', 'jessica_lange',
     'jim_calhoun', 'jim_flaherty', 'jim_furyk', 'jimmy_smits', 'joan_laporta', 'joaquin_phoenix', 'jodie_foster',
     'jodie_kidd', 'joe_calzaghe', 'johan_bruyneel', 'john_cornyn', 'john_cusack', 'john_elway', 'john_kerry',
     'john_leguizamo', 'john_lithgow', 'john_travolta', 'johnny_depp', 'jon_voight', 'joseph_estrada', 'joseph_fiennes',
     'joy_bryant', 'jude_law', 'judi_dench', 'julia_ormond', 'julianna_margulies', 'julie_taymor', 'juliette_binoche',
     'justin_gatlin', 'justin_guarini', 'justine_henin', 'karin_stoiber', 'karl-heinz_rummenigge', 'kate_hudson',
     'kathryn_bigelow', 'kathryn_morris', 'katie_couric', 'katie_holmes', 'katja_riemann', 'kay_bailey_hutchison',
     'kelly_ripa', 'ken_loach', 'kenenisa_bekele', 'kevin_costner', 'kevin_sorbo', 'kim_cattrall', 'kim_clijsters',
     'kirk_franklin', 'kirsten_dunst', 'kobe_bryant', 'kristanna_loken', 'kristin_davis', 'kristin_scott_thomas',
     'kyra_sedgwick', 'larenz_tate', 'laura_linney', 'laurence_fishburne', 'laurent_jalabert', 'leah_remini',
     'leander_paes', 'leann_rimes', 'lebron_james', 'lee_ann_womack', 'lela_rochon', 'lena_olin', 'lene_espersen',
     'lennox_lewis', 'leon_lai', 'leonardo_dicaprio', 'leslie_moonves', 'leticia_van_de_putte', 'liam_neeson',
     'linus_roache', 'lisa_leslie', 'lisa_ling', 'lisa_marie_presley', 'lisa_murkowski', 'lisa_stansfield', 'liv_tyler',
     'liza_minnelli', 'lleyton_hewitt', 'lorne_michaels', 'louis_van_gaal', 'luca_cordero_di_montezemolo',
     'madeleine_albright', 'maggie_cheung', 'manuel_pellegrini', 'marcelo_salas', 'maria_bello', 'marina_hands',
     'marion_barry', 'marisa_tomei', 'marissa_jaret_winokur', 'mark_leno', 'martie_maguire', 'martin_oneill',
     'martina_mcbride', 'mary_carey', 'mary_elizabeth_mastrantonio', 'mary_steenburgen', 'matt_damon', 'matthew_perry',
     'matthias_sammer', 'maura_tierney', 'max_mosley', 'mekhi_phifer', 'melissa_etheridge', 'melissa_gilbert',
     'melissa_joan_hart', 'melissa_manchester', 'michael_ballack', 'michael_bloomberg', 'michael_chiklis',
     'michael_jackson', 'michael_keaton', 'michael_michele', 'michael_phelps', 'michael_schumacher', 'michel_temer',
     'michel_therrien', 'michelle_collins', 'michelle_rodriguez', 'mick_jagger', 'miguel_cotto', 'mike_brey',
     'mike_matheny', 'mike_miller', 'mike_tyson', 'minnie_driver', 'mira_sorvino', 'miranda_otto', 'molly_sims',
     'monica_lewinsky', 'monica_seles', 'morgan_fairchild', 'muffet_mcgraw', 'nancy_kerrigan', 'nancy_pelosi',
     'naomi_watts', 'narendra_modi', 'natalia_verbeke', 'natalie_coughlin', 'natalie_imbruglia', 'natalie_maines',
     'natasha_henstridge', 'nathalie_baye', 'nathan_lane', 'nicolas_kiefer', 'noah_wyle', 'nona_gaye', 'oliver_stone',
     'oprah_winfrey', 'orlando_bloom', 'ornella_muti', 'oscar_de_la_hoya', 'owen_wilson', 'ozzy_osbourne',
     'pascal_lamy', 'pat_summitt', 'patsy_kensit', 'patti_labelle', 'paul_greengrass', 'paul_mccartney', 'paula_abdul',
     'paula_zahn', 'penelope_ann_miller', 'penny_lancaster', 'percy_gibson', 'perry_farrell', 'pete_carroll',
     'pete_rose', 'phil_jackson', 'phil_mcgraw', 'phil_mickelson', 'pier_ferdinando_casini', 'pierre_van_hooijdonk',
     'pilar_montenegro', 'priscilla_presley', 'rachel_griffiths', 'rachel_hunter', 'rafidah_aziz', 'rahul_dravid',
     'ralf_schumacher', 'ralph_klein', 'randy_travis', 'raquel_welch', 'ray_romano', 'rena_sofer', 'retief_goosen',
     'ricardo_mayorga', 'riccardo_muti', 'richard_armitage', 'rick_carlisle', 'rick_perry', 'rick_pitino',
     'ricky_ponting', 'rio_ferdinand', 'rita_wilson', 'rob_lowe', 'rob_schneider', 'robert_de_niro', 'robert_redford',
     'robin_tunney', 'rod_stewart', 'roger_federer', 'rolandas_paksas', 'romain_duris', 'roseanne_barr', 'rosie_perez',
     'rowan_williams', 'roy_blunt', 'ruben_studdard', 'rubens_barrichello', 'rulon_gardner', 'rupert_grint',
     'russell_crowe', 'russell_simmons', 'ryan_newman', 'sadie_frost', 'sam_brownback', 'sam_rockwell',
     'sarah_jessica_parker', 'sarah_wynter', 'sasha_alexander', 'sean_hayes', 'sepp_blatter', 'shannyn_sossamon',
     'sharon_osbourne', 'sharon_stone', 'sherri_coale', 'sigourney_weaver', 'simon_cowell', 'sofia_milos',
     'sonya_walger', 'sophia_loren', 'sourav_ganguly', 'stefano_accorsi', 'stella_tennant', 'steve_ballmer',
     'steve_coogan', 'steven_van_zandt', 'stockard_channing', 'susan_sarandon', 'suzanne_somers', 'sylvester_stallone',
     'tatjana_gsell', 'taufik_hidayat', 'ted_nolan', 'teri_garr', 'theo_epstein', 'thomas_gottschalk', 'tia_mowry',
     'tiger_woods', 'tim_duncan', 'tim_howard', 'tina_fey', 'tirunesh_dibaba', 'toby_keith', 'tom_vilsack',
     'toni_braxton', 'tony_fernandes', 'tony_shalhoub', 'tony_stewart', 'tracy_mcgrady', 'trevor_mcdonald',
     'troy_garity', 'troy_polamalu', 'tyler_hamilton', 'valentina_cervi', 'valerie_harper', 'vanessa_incontrada',
     'venus_williams', 'vicente_fox', 'victor_garber', 'viktor_yushchenko', 'vince_gill', 'viola_davis',
     'vitali_klitschko', 'whoopi_goldberg', 'will_young', 'xavier_malisse', 'yuvraj_singh', 'zinedine_zidane']
root = "/nfs/nas4/marzieh/marzieh/VGG_Face2/lfw/lfw-deepfunneled/"
for i, item in enumerate(a):
    os.system("rm -r " + root + item.title())
