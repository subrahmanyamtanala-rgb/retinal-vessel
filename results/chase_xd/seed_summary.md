### chase_xd scale 0.55

| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |
|---|---|---|---|---|
| unet | 3 | 0.7688 ± 0.0019 (0.7667..0.7702) | 0.9813 ± 0.0003 | 0.8423 ± 0.0019 |
| no-router | 2 | 0.7644 ± 0.0041 (0.7615..0.7673) | 0.9800 ± 0.0005 | 0.8403 ± 0.0000 |
| catnet-s | 1 | 0.7678 ± 0.0000 | 0.9802 ± 0.0000 | 0.8424 ± 0.0000 |
| ac-full | 2 | 0.7661 ± 0.0034 (0.7637..0.7685) | 0.9802 ± 0.0000 | 0.8432 ± 0.0008 |
| q-norouter | 1 | 0.7683 ± 0.0000 | 0.9801 ± 0.0000 | 0.8425 ± 0.0000 |
| q-full | 1 | 0.7655 ± 0.0000 | 0.9807 ± 0.0000 | 0.8407 ± 0.0000 |
| unet-saug | 2 | 0.7718 ± 0.0010 (0.7711..0.7725) | 0.9818 ± 0.0001 | 0.8420 ± 0.0020 |
| no-router-saug | 2 | 0.7650 ± 0.0017 (0.7638..0.7662) | 0.9801 ± 0.0002 | 0.8355 ± 0.0053 |
| ac-full-saug | 2 | 0.7656 ± 0.0008 (0.7650..0.7662) | 0.9804 ± 0.0002 | 0.8345 ± 0.0001 |

| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |
|---|---|---|---|---|---|
| no-router − unet | f1 | -0.0044 | 4/28 | 6.3e-05 | no (-0.0087..+0.0006) |
| no-router − unet | auc | -0.0013 | 2/28 | 8.2e-05 | yes (-0.0019..-0.0007) |
| no-router − unet | cldice | -0.0020 | 9/28 | 0.095 | no (-0.0034..+0.0002) |
| catnet-s − unet | f1 | -0.0011 | 8/28 | 0.11 | no (-0.0024..+0.0011) |
| catnet-s − unet | auc | -0.0011 | 3/28 | 6.3e-05 | yes (-0.0014..-0.0008) |
| catnet-s − unet | cldice | +0.0001 | 15/28 | 0.85 | no (-0.0013..+0.0023) |
| ac-full − unet | f1 | -0.0027 | 6/28 | 0.0034 | no (-0.0065..+0.0019) |
| ac-full − unet | auc | -0.0011 | 3/28 | 4.2e-05 | yes (-0.0014..-0.0009) |
| ac-full − unet | cldice | +0.0010 | 17/28 | 0.14 | no (-0.0010..+0.0037) |
| q-norouter − unet | f1 | -0.0005 | 11/28 | 0.33 | no (-0.0019..+0.0016) |
| q-norouter − unet | auc | -0.0012 | 3/28 | 0.00053 | yes (-0.0015..-0.0009) |
| q-norouter − unet | cldice | +0.0002 | 13/28 | 0.83 | no (-0.0013..+0.0024) |
| q-full − unet | f1 | -0.0033 | 4/28 | 0.00015 | yes (-0.0047..-0.0012) |
| q-full − unet | auc | -0.0006 | 4/28 | 0.0032 | yes (-0.0009..-0.0004) |
| q-full − unet | cldice | -0.0016 | 10/28 | 0.11 | no (-0.0030..+0.0006) |
| ac-full − no-router | f1 | +0.0017 | 24/28 | 1.1e-05 | no (-0.0036..+0.0071) |
| ac-full − no-router | auc | +0.0002 | 19/28 | 0.032 | no (-0.0002..+0.0005) |
| ac-full − no-router | cldice | +0.0029 | 25/28 | 1.5e-05 | yes (+0.0024..+0.0035) |
| catnet-s − no-router | f1 | +0.0034 | 27/28 | 1.5e-08 | yes (+0.0004..+0.0063) |
| catnet-s − no-router | auc | +0.0002 | 18/28 | 0.017 | no (-0.0001..+0.0005) |
| catnet-s − no-router | cldice | +0.0021 | 19/28 | 0.0037 | yes (+0.0020..+0.0021) |
| q-full − q-norouter | f1 | -0.0028 | 4/28 | 5.5e-05 | yes (-0.0028..-0.0028) |
| q-full − q-norouter | auc | +0.0006 | 23/28 | 0.0022 | yes (+0.0006..+0.0006) |
| q-full − q-norouter | cldice | -0.0018 | 7/28 | 0.0076 | yes (-0.0018..-0.0018) |
| unet-saug − unet | f1 | +0.0030 | 24/28 | 9.3e-05 | yes (+0.0009..+0.0059) |
| unet-saug − unet | auc | +0.0005 | 16/28 | 0.29 | yes (+0.0001..+0.0008) |
| unet-saug − unet | cldice | -0.0003 | 11/28 | 0.69 | no (-0.0032..+0.0033) |
| no-router-saug − no-router | f1 | +0.0006 | 18/28 | 0.37 | no (-0.0035..+0.0047) |
| no-router-saug − no-router | auc | +0.0001 | 14/28 | 0.8 | no (-0.0004..+0.0006) |
| no-router-saug − no-router | cldice | -0.0048 | 7/28 | 0.00072 | yes (-0.0086..-0.0010) |
| ac-full-saug − ac-full | f1 | -0.0005 | 15/28 | 0.99 | no (-0.0035..+0.0025) |
| ac-full-saug − ac-full | auc | +0.0002 | 13/28 | 0.8 | yes (+0.0001..+0.0003) |
| ac-full-saug − ac-full | cldice | -0.0087 | 3/28 | 1.3e-05 | yes (-0.0093..-0.0081) |
| no-router-saug − unet-saug | f1 | -0.0068 | 1/28 | 1.5e-08 | yes (-0.0087..-0.0049) |
| no-router-saug − unet-saug | auc | -0.0017 | 1/28 | 1.5e-08 | yes (-0.0020..-0.0014) |
| no-router-saug − unet-saug | cldice | -0.0065 | 2/28 | 7.5e-08 | yes (-0.0116..-0.0013) |
| ac-full-saug − unet-saug | f1 | -0.0062 | 1/28 | 1.5e-08 | yes (-0.0075..-0.0049) |
| ac-full-saug − unet-saug | auc | -0.0014 | 1/28 | 3.7e-08 | yes (-0.0016..-0.0013) |
| ac-full-saug − unet-saug | cldice | -0.0075 | 2/28 | 1.9e-07 | yes (-0.0089..-0.0060) |
| ac-full-saug − no-router-saug | f1 | +0.0006 | 16/28 | 0.14 | no (-0.0011..+0.0024) |
| ac-full-saug − no-router-saug | auc | +0.0003 | 21/28 | 0.002 | no (-0.0000..+0.0005) |
| ac-full-saug − no-router-saug | cldice | -0.0010 | 12/28 | 0.27 | no (-0.0048..+0.0028) |

### chase_xd scale 0.75

| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |
|---|---|---|---|---|
| unet | 3 | 0.7437 ± 0.0046 (0.7409..0.7490) | 0.9711 ± 0.0010 | 0.7726 ± 0.0079 |
| no-router | 2 | 0.7506 ± 0.0017 (0.7494..0.7518) | 0.9714 ± 0.0003 | 0.7869 ± 0.0039 |
| catnet-s | 1 | 0.7484 ± 0.0000 | 0.9716 ± 0.0000 | 0.7851 ± 0.0000 |
| ac-full | 2 | 0.7466 ± 0.0061 (0.7423..0.7509) | 0.9700 ± 0.0009 | 0.7850 ± 0.0032 |
| q-norouter | 1 | 0.7550 ± 0.0000 | 0.9722 ± 0.0000 | 0.7900 ± 0.0000 |
| q-full | 1 | 0.7516 ± 0.0000 | 0.9718 ± 0.0000 | 0.7905 ± 0.0000 |
| unet-saug | 2 | 0.7761 ± 0.0002 (0.7760..0.7763) | 0.9797 ± 0.0002 | 0.8299 ± 0.0006 |
| no-router-saug | 2 | 0.7704 ± 0.0025 (0.7686..0.7722) | 0.9780 ± 0.0003 | 0.8206 ± 0.0036 |
| ac-full-saug | 2 | 0.7690 ± 0.0018 (0.7677..0.7703) | 0.9779 ± 0.0001 | 0.8174 ± 0.0013 |

| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |
|---|---|---|---|---|---|
| no-router − unet | f1 | +0.0069 | 20/28 | 0.00079 | yes (+0.0004..+0.0108) |
| no-router − unet | auc | +0.0003 | 14/28 | 0.88 | no (-0.0008..+0.0016) |
| no-router − unet | cldice | +0.0142 | 26/28 | 1e-07 | yes (+0.0025..+0.0230) |
| catnet-s − unet | f1 | +0.0048 | 20/28 | 0.0048 | no (-0.0005..+0.0075) |
| catnet-s − unet | auc | +0.0005 | 16/28 | 0.48 | no (-0.0003..+0.0016) |
| catnet-s − unet | cldice | +0.0125 | 24/28 | 5.2e-07 | yes (+0.0035..+0.0185) |
| ac-full − unet | f1 | +0.0030 | 15/28 | 0.35 | no (-0.0067..+0.0100) |
| ac-full − unet | auc | -0.0011 | 9/28 | 0.022 | no (-0.0026..+0.0006) |
| ac-full − unet | cldice | +0.0123 | 25/28 | 4.8e-06 | yes (+0.0011..+0.0206) |
| q-norouter − unet | f1 | +0.0114 | 24/28 | 1.3e-06 | yes (+0.0060..+0.0141) |
| q-norouter − unet | auc | +0.0011 | 17/28 | 0.11 | yes (+0.0003..+0.0022) |
| q-norouter − unet | cldice | +0.0173 | 27/28 | 1.5e-08 | yes (+0.0084..+0.0233) |
| q-full − unet | f1 | +0.0079 | 21/28 | 0.00097 | yes (+0.0026..+0.0106) |
| q-full − unet | auc | +0.0007 | 13/28 | 0.65 | no (-0.0002..+0.0017) |
| q-full − unet | cldice | +0.0178 | 26/28 | 1.4e-07 | yes (+0.0089..+0.0238) |
| ac-full − no-router | f1 | -0.0040 | 5/28 | 0.00034 | no (-0.0095..+0.0015) |
| ac-full − no-router | auc | -0.0014 | 4/28 | 2.8e-06 | yes (-0.0023..-0.0006) |
| ac-full − no-router | cldice | -0.0019 | 11/28 | 0.086 | no (-0.0069..+0.0032) |
| catnet-s − no-router | f1 | -0.0022 | 13/28 | 0.13 | yes (-0.0033..-0.0010) |
| catnet-s − no-router | auc | +0.0002 | 16/28 | 0.32 | no (-0.0001..+0.0004) |
| catnet-s − no-router | cldice | -0.0017 | 11/28 | 0.23 | no (-0.0045..+0.0011) |
| q-full − q-norouter | f1 | -0.0035 | 8/28 | 0.0024 | yes (-0.0035..-0.0035) |
| q-full − q-norouter | auc | -0.0004 | 9/28 | 0.066 | yes (-0.0004..-0.0004) |
| q-full − q-norouter | cldice | +0.0005 | 19/28 | 0.48 | yes (+0.0005..+0.0005) |
| unet-saug − unet | f1 | +0.0325 | 25/28 | 3.2e-07 | yes (+0.0270..+0.0353) |
| unet-saug − unet | auc | +0.0086 | 28/28 | 7.5e-09 | yes (+0.0077..+0.0098) |
| unet-saug − unet | cldice | +0.0572 | 28/28 | 7.5e-09 | yes (+0.0478..+0.0637) |
| no-router-saug − no-router | f1 | +0.0198 | 24/28 | 2.8e-06 | yes (+0.0168..+0.0228) |
| no-router-saug − no-router | auc | +0.0066 | 28/28 | 7.5e-09 | yes (+0.0061..+0.0071) |
| no-router-saug − no-router | cldice | +0.0337 | 28/28 | 7.5e-09 | yes (+0.0283..+0.0391) |
| ac-full-saug − ac-full | f1 | +0.0224 | 24/28 | 9.4e-06 | yes (+0.0168..+0.0280) |
| ac-full-saug − ac-full | auc | +0.0080 | 28/28 | 7.5e-09 | yes (+0.0072..+0.0087) |
| ac-full-saug − ac-full | cldice | +0.0324 | 27/28 | 1.5e-08 | yes (+0.0292..+0.0356) |
| no-router-saug − unet-saug | f1 | -0.0058 | 2/28 | 5.2e-08 | yes (-0.0077..-0.0038) |
| no-router-saug − unet-saug | auc | -0.0017 | 0/28 | 7.5e-09 | yes (-0.0020..-0.0014) |
| no-router-saug − unet-saug | cldice | -0.0093 | 1/28 | 2.2e-08 | yes (-0.0123..-0.0063) |
| ac-full-saug − unet-saug | f1 | -0.0071 | 1/28 | 2.2e-08 | yes (-0.0085..-0.0057) |
| ac-full-saug − unet-saug | auc | -0.0018 | 1/28 | 1.5e-08 | yes (-0.0020..-0.0016) |
| ac-full-saug − unet-saug | cldice | -0.0125 | 0/28 | 7.5e-09 | yes (-0.0139..-0.0111) |
| ac-full-saug − no-router-saug | f1 | -0.0014 | 11/28 | 0.053 | no (-0.0044..+0.0017) |
| ac-full-saug − no-router-saug | auc | -0.0001 | 13/28 | 0.54 | no (-0.0004..+0.0002) |
| ac-full-saug − no-router-saug | cldice | -0.0032 | 9/28 | 0.00034 | no (-0.0067..+0.0003) |

### chase_xd scale 1

| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |
|---|---|---|---|---|
| unet | 3 | 0.5064 ± 0.0363 (0.4673..0.5389) | 0.9271 ± 0.0086 | 0.4778 ± 0.0492 |
| no-router | 2 | 0.6097 ± 0.0010 (0.6090..0.6104) | 0.9422 ± 0.0025 | 0.6053 ± 0.0138 |
| catnet-s | 1 | 0.5899 ± 0.0000 | 0.9414 ± 0.0000 | 0.5926 ± 0.0000 |
| ac-full | 2 | 0.5775 ± 0.0294 (0.5567..0.5983) | 0.9269 ± 0.0089 | 0.5774 ± 0.0147 |
| q-norouter | 1 | 0.5830 ± 0.0000 | 0.9424 ± 0.0000 | 0.5724 ± 0.0000 |
| q-full | 1 | 0.5823 ± 0.0000 | 0.9353 ± 0.0000 | 0.5848 ± 0.0000 |
| unet-saug | 2 | 0.7576 ± 0.0014 (0.7566..0.7585) | 0.9740 ± 0.0006 | 0.7852 ± 0.0013 |
| no-router-saug | 2 | 0.7540 ± 0.0015 (0.7530..0.7550) | 0.9724 ± 0.0002 | 0.7757 ± 0.0036 |
| ac-full-saug | 2 | 0.7524 ± 0.0017 (0.7512..0.7536) | 0.9720 ± 0.0004 | 0.7732 ± 0.0019 |

| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |
|---|---|---|---|---|---|
| no-router − unet | f1 | +0.1033 | 28/28 | 7.5e-09 | yes (+0.0701..+0.1431) |
| no-router − unet | auc | +0.0151 | 28/28 | 7.5e-09 | yes (+0.0054..+0.0260) |
| no-router − unet | cldice | +0.1275 | 28/28 | 7.5e-09 | yes (+0.0712..+0.1887) |
| catnet-s − unet | f1 | +0.0835 | 28/28 | 7.5e-09 | yes (+0.0510..+0.1226) |
| catnet-s − unet | auc | +0.0143 | 28/28 | 7.5e-09 | yes (+0.0064..+0.0234) |
| catnet-s − unet | cldice | +0.1149 | 28/28 | 7.5e-09 | yes (+0.0683..+0.1663) |
| ac-full − unet | f1 | +0.0710 | 26/28 | 3.7e-08 | yes (+0.0178..+0.1310) |
| ac-full − unet | auc | -0.0002 | 12/28 | 0.71 | no (-0.0144..+0.0151) |
| ac-full − unet | cldice | +0.0996 | 27/28 | 1.5e-08 | yes (+0.0427..+0.1615) |
| q-norouter − unet | f1 | +0.0765 | 28/28 | 7.5e-09 | yes (+0.0441..+0.1157) |
| q-norouter − unet | auc | +0.0152 | 28/28 | 7.5e-09 | yes (+0.0074..+0.0244) |
| q-norouter − unet | cldice | +0.0947 | 28/28 | 7.5e-09 | yes (+0.0481..+0.1461) |
| q-full − unet | f1 | +0.0759 | 28/28 | 7.5e-09 | yes (+0.0435..+0.1150) |
| q-full − unet | auc | +0.0082 | 24/28 | 5.5e-05 | yes (+0.0003..+0.0173) |
| q-full − unet | cldice | +0.1070 | 28/28 | 7.5e-09 | yes (+0.0605..+0.1585) |
| ac-full − no-router | f1 | -0.0322 | 0/28 | 7.5e-09 | yes (-0.0537..-0.0107) |
| ac-full − no-router | auc | -0.0153 | 0/28 | 7.5e-09 | yes (-0.0234..-0.0073) |
| ac-full − no-router | cldice | -0.0279 | 0/28 | 7.5e-09 | yes (-0.0480..-0.0078) |
| catnet-s − no-router | f1 | -0.0198 | 1/28 | 2.2e-08 | yes (-0.0205..-0.0191) |
| catnet-s − no-router | auc | -0.0008 | 12/28 | 0.34 | no (-0.0025..+0.0010) |
| catnet-s − no-router | cldice | -0.0126 | 3/28 | 6.6e-07 | yes (-0.0224..-0.0029) |
| q-full − q-norouter | f1 | -0.0006 | 13/28 | 0.9 | yes (-0.0006..-0.0006) |
| q-full − q-norouter | auc | -0.0071 | 1/28 | 2.2e-08 | yes (-0.0071..-0.0071) |
| q-full − q-norouter | cldice | +0.0124 | 23/28 | 0.00058 | yes (+0.0124..+0.0124) |
| unet-saug − unet | f1 | +0.2511 | 28/28 | 7.5e-09 | yes (+0.2177..+0.2912) |
| unet-saug − unet | auc | +0.0469 | 28/28 | 7.5e-09 | yes (+0.0386..+0.0563) |
| unet-saug − unet | cldice | +0.3074 | 28/28 | 7.5e-09 | yes (+0.2600..+0.3599) |
| no-router-saug − no-router | f1 | +0.1443 | 28/28 | 7.5e-09 | yes (+0.1425..+0.1460) |
| no-router-saug − no-router | auc | +0.0302 | 28/28 | 7.5e-09 | yes (+0.0283..+0.0321) |
| no-router-saug − no-router | cldice | +0.1704 | 28/28 | 7.5e-09 | yes (+0.1581..+0.1827) |
| ac-full-saug − ac-full | f1 | +0.1749 | 28/28 | 7.5e-09 | yes (+0.1529..+0.1968) |
| ac-full-saug − ac-full | auc | +0.0452 | 28/28 | 7.5e-09 | yes (+0.0386..+0.0517) |
| ac-full-saug − ac-full | cldice | +0.1958 | 28/28 | 7.5e-09 | yes (+0.1841..+0.2075) |
| no-router-saug − unet-saug | f1 | -0.0036 | 6/28 | 0.00038 | yes (-0.0056..-0.0016) |
| no-router-saug − unet-saug | auc | -0.0016 | 3/28 | 3.2e-07 | yes (-0.0021..-0.0010) |
| no-router-saug − unet-saug | cldice | -0.0096 | 1/28 | 3.7e-08 | yes (-0.0131..-0.0061) |
| ac-full-saug − unet-saug | f1 | -0.0052 | 5/28 | 0.00015 | yes (-0.0073..-0.0030) |
| ac-full-saug − unet-saug | auc | -0.0019 | 1/28 | 6.6e-07 | yes (-0.0026..-0.0013) |
| ac-full-saug − unet-saug | cldice | -0.0120 | 2/28 | 1e-07 | yes (-0.0143..-0.0097) |
| ac-full-saug − no-router-saug | f1 | -0.0016 | 12/28 | 0.24 | no (-0.0038..+0.0006) |
| ac-full-saug − no-router-saug | auc | -0.0004 | 6/28 | 0.011 | no (-0.0008..+0.0001) |
| ac-full-saug − no-router-saug | cldice | -0.0024 | 9/28 | 0.03 | no (-0.0063..+0.0015) |
