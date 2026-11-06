### hrf_xd scale 0.35

| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |
|---|---|---|---|---|
| unet | 3 | 0.7578 ± 0.0015 (0.7562..0.7591) | 0.9834 ± 0.0002 | 0.8364 ± 0.0023 |
| no-router | 2 | 0.7570 ± 0.0066 (0.7523..0.7616) | 0.9817 ± 0.0006 | 0.8354 ± 0.0001 |
| catnet-s | 1 | 0.7594 ± 0.0000 | 0.9823 ± 0.0000 | 0.8372 ± 0.0000 |
| ac-full | 2 | 0.7590 ± 0.0020 (0.7576..0.7604) | 0.9823 ± 0.0000 | 0.8370 ± 0.0010 |
| q-norouter | 1 | 0.7606 ± 0.0000 | 0.9824 ± 0.0000 | 0.8354 ± 0.0000 |
| q-full | 1 | 0.7571 ± 0.0000 | 0.9831 ± 0.0000 | 0.8351 ± 0.0000 |
| unet-saug | 2 | 0.7585 ± 0.0011 (0.7577..0.7593) | 0.9840 ± 0.0001 | 0.8350 ± 0.0024 |
| no-router-saug | 2 | 0.7563 ± 0.0038 (0.7536..0.7590) | 0.9825 ± 0.0002 | 0.8322 ± 0.0002 |
| ac-full-saug | 2 | 0.7562 ± 0.0029 (0.7541..0.7583) | 0.9832 ± 0.0002 | 0.8335 ± 0.0003 |

| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |
|---|---|---|---|---|---|
| no-router − unet | f1 | -0.0009 | 15/45 | 0.025 | no (-0.0068..+0.0054) |
| no-router − unet | auc | -0.0017 | 2/45 | 4e-13 | yes (-0.0023..-0.0010) |
| no-router − unet | cldice | -0.0010 | 14/45 | 0.0055 | no (-0.0025..+0.0017) |
| catnet-s − unet | f1 | +0.0016 | 33/45 | 0.00021 | yes (+0.0003..+0.0032) |
| catnet-s − unet | auc | -0.0010 | 5/45 | 1.2e-09 | yes (-0.0012..-0.0008) |
| catnet-s − unet | cldice | +0.0007 | 26/45 | 0.16 | no (-0.0007..+0.0034) |
| ac-full − unet | f1 | +0.0012 | 32/45 | 0.0012 | no (-0.0015..+0.0042) |
| ac-full − unet | auc | -0.0011 | 1/45 | 1.9e-12 | yes (-0.0013..-0.0009) |
| ac-full − unet | cldice | +0.0006 | 25/45 | 0.31 | no (-0.0016..+0.0039) |
| q-norouter − unet | f1 | +0.0028 | 36/45 | 3.8e-07 | yes (+0.0015..+0.0044) |
| q-norouter − unet | auc | -0.0010 | 6/45 | 2.3e-08 | yes (-0.0011..-0.0008) |
| q-norouter − unet | cldice | -0.0010 | 15/45 | 0.038 | no (-0.0025..+0.0016) |
| q-full − unet | f1 | -0.0007 | 20/45 | 0.23 | no (-0.0020..+0.0009) |
| q-full − unet | auc | -0.0003 | 11/45 | 0.0022 | yes (-0.0004..-0.0001) |
| q-full − unet | cldice | -0.0013 | 14/45 | 0.011 | no (-0.0027..+0.0013) |
| ac-full − no-router | f1 | +0.0021 | 39/45 | 6.4e-09 | no (-0.0040..+0.0081) |
| ac-full − no-router | auc | +0.0006 | 32/45 | 1e-05 | yes (+0.0001..+0.0010) |
| ac-full − no-router | cldice | +0.0016 | 35/45 | 2.7e-05 | yes (+0.0008..+0.0024) |
| catnet-s − no-router | f1 | +0.0025 | 40/45 | 1.5e-07 | no (-0.0022..+0.0072) |
| catnet-s − no-router | auc | +0.0006 | 38/45 | 6.8e-07 | yes (+0.0002..+0.0011) |
| catnet-s − no-router | cldice | +0.0017 | 34/45 | 2.8e-05 | yes (+0.0017..+0.0018) |
| q-full − q-norouter | f1 | -0.0035 | 7/45 | 6.8e-07 | yes (-0.0035..-0.0035) |
| q-full − q-norouter | auc | +0.0007 | 37/45 | 5.8e-07 | yes (+0.0007..+0.0007) |
| q-full − q-norouter | cldice | -0.0003 | 22/45 | 0.97 | yes (-0.0003..-0.0003) |
| unet-saug − unet | f1 | +0.0006 | 32/45 | 0.057 | no (-0.0014..+0.0030) |
| unet-saug − unet | auc | +0.0007 | 31/45 | 0.00021 | yes (+0.0004..+0.0010) |
| unet-saug − unet | cldice | -0.0015 | 16/45 | 0.019 | no (-0.0046..+0.0028) |
| no-router-saug − no-router | f1 | -0.0006 | 26/45 | 0.96 | no (-0.0080..+0.0068) |
| no-router-saug − no-router | auc | +0.0008 | 33/45 | 0.00014 | yes (+0.0002..+0.0014) |
| no-router-saug − no-router | cldice | -0.0032 | 12/45 | 0.00016 | yes (-0.0035..-0.0030) |
| ac-full-saug − ac-full | f1 | -0.0028 | 9/45 | 5.7e-06 | no (-0.0063..+0.0007) |
| ac-full-saug − ac-full | auc | +0.0009 | 37/45 | 9.6e-08 | yes (+0.0008..+0.0010) |
| ac-full-saug − ac-full | cldice | -0.0035 | 6/45 | 1.8e-07 | yes (-0.0044..-0.0025) |
| no-router-saug − unet-saug | f1 | -0.0021 | 12/45 | 0.00023 | no (-0.0056..+0.0014) |
| no-router-saug − unet-saug | auc | -0.0016 | 0/45 | 5.7e-14 | yes (-0.0018..-0.0013) |
| no-router-saug − unet-saug | cldice | -0.0028 | 9/45 | 4.5e-07 | yes (-0.0046..-0.0010) |
| ac-full-saug − unet-saug | f1 | -0.0023 | 6/45 | 6.3e-07 | no (-0.0051..+0.0006) |
| ac-full-saug − unet-saug | auc | -0.0009 | 0/45 | 5.7e-14 | yes (-0.0011..-0.0007) |
| ac-full-saug − unet-saug | cldice | -0.0015 | 12/45 | 0.00027 | no (-0.0033..+0.0004) |
| ac-full-saug − no-router-saug | f1 | -0.0001 | 22/45 | 0.7 | no (-0.0049..+0.0047) |
| ac-full-saug − no-router-saug | auc | +0.0007 | 42/45 | 9.6e-12 | yes (+0.0004..+0.0010) |
| ac-full-saug − no-router-saug | cldice | +0.0013 | 30/45 | 0.00013 | yes (+0.0010..+0.0017) |

### hrf_xd scale 0.5

| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |
|---|---|---|---|---|
| unet | 1 | 0.7776 ± 0.0000 | 0.9761 ± 0.0000 | 0.8246 ± 0.0000 |
| no-router | 1 | 0.7786 ± 0.0000 | 0.9747 ± 0.0000 | 0.8240 ± 0.0000 |
| ac-full | 1 | 0.7708 ± 0.0000 | 0.9724 ± 0.0000 | 0.8227 ± 0.0000 |
| q-full | 1 | 0.7732 ± 0.0000 | 0.9771 ± 0.0000 | 0.8260 ± 0.0000 |

| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |
|---|---|---|---|---|---|
| no-router − unet | f1 | +0.0010 | 29/45 | 0.11 | yes (+0.0010..+0.0010) |
| no-router − unet | auc | -0.0014 | 11/45 | 0.00068 | yes (-0.0014..-0.0014) |
| no-router − unet | cldice | -0.0006 | 19/45 | 0.18 | yes (-0.0006..-0.0006) |
| ac-full − unet | f1 | -0.0067 | 6/45 | 1.2e-09 | yes (-0.0067..-0.0067) |
| ac-full − unet | auc | -0.0037 | 8/45 | 1.5e-07 | yes (-0.0037..-0.0037) |
| ac-full − unet | cldice | -0.0019 | 14/45 | 0.002 | yes (-0.0019..-0.0019) |
| q-full − unet | f1 | -0.0044 | 12/45 | 4.1e-05 | yes (-0.0044..-0.0044) |
| q-full − unet | auc | +0.0010 | 29/45 | 0.032 | yes (+0.0010..+0.0010) |
| q-full − unet | cldice | +0.0014 | 24/45 | 0.24 | yes (+0.0014..+0.0014) |
| ac-full − no-router | f1 | -0.0077 | 3/45 | 7.8e-12 | yes (-0.0077..-0.0077) |
| ac-full − no-router | auc | -0.0023 | 18/45 | 0.0088 | yes (-0.0023..-0.0023) |
| ac-full − no-router | cldice | -0.0013 | 24/45 | 0.25 | yes (-0.0013..-0.0013) |

### hrf_xd scale 0.7

| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |
|---|---|---|---|---|
| unet | 3 | 0.7057 ± 0.0129 (0.6908..0.7146) | 0.9534 ± 0.0046 | 0.7381 ± 0.0044 |
| no-router | 2 | 0.7068 ± 0.0068 (0.7020..0.7116) | 0.9470 ± 0.0010 | 0.7417 ± 0.0010 |
| ac-full | 2 | 0.6946 ± 0.0222 (0.6789..0.7103) | 0.9336 ± 0.0117 | 0.7312 ± 0.0108 |
| unet-saug | 2 | 0.7678 ± 0.0035 (0.7653..0.7702) | 0.9802 ± 0.0013 | 0.8122 ± 0.0061 |
| no-router-saug | 2 | 0.7673 ± 0.0000 (0.7673..0.7673) | 0.9774 ± 0.0014 | 0.8137 ± 0.0033 |
| ac-full-saug | 2 | 0.7675 ± 0.0016 (0.7664..0.7686) | 0.9788 ± 0.0008 | 0.8127 ± 0.0008 |

| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |
|---|---|---|---|---|---|
| no-router − unet | f1 | +0.0011 | 24/45 | 0.56 | no (-0.0126..+0.0208) |
| no-router − unet | auc | -0.0064 | 7/45 | 1.8e-09 | yes (-0.0124..-0.0028) |
| no-router − unet | cldice | +0.0036 | 31/45 | 0.0085 | yes (+0.0003..+0.0094) |
| ac-full − unet | f1 | -0.0111 | 7/45 | 4.9e-10 | no (-0.0357..+0.0194) |
| ac-full − unet | auc | -0.0198 | 0/45 | 5.7e-14 | yes (-0.0333..-0.0086) |
| ac-full − unet | cldice | -0.0069 | 11/45 | 1.5e-05 | no (-0.0172..+0.0058) |
| ac-full − no-router | f1 | -0.0122 | 1/45 | 1.1e-13 | no (-0.0327..+0.0083) |
| ac-full − no-router | auc | -0.0134 | 3/45 | 7.8e-12 | yes (-0.0223..-0.0044) |
| ac-full − no-router | cldice | -0.0106 | 1/45 | 1.7e-13 | yes (-0.0189..-0.0022) |
| unet-saug − unet | f1 | +0.0621 | 45/45 | 5.7e-14 | yes (+0.0507..+0.0794) |
| unet-saug − unet | auc | +0.0268 | 45/45 | 5.7e-14 | yes (+0.0206..+0.0306) |
| unet-saug − unet | cldice | +0.0741 | 45/45 | 5.7e-14 | yes (+0.0672..+0.0835) |
| no-router-saug − no-router | f1 | +0.0605 | 45/45 | 5.7e-14 | yes (+0.0557..+0.0654) |
| no-router-saug − no-router | auc | +0.0304 | 45/45 | 5.7e-14 | yes (+0.0287..+0.0320) |
| no-router-saug − no-router | cldice | +0.0720 | 45/45 | 5.7e-14 | yes (+0.0690..+0.0751) |
| ac-full-saug − ac-full | f1 | +0.0729 | 45/45 | 5.7e-14 | yes (+0.0561..+0.0897) |
| ac-full-saug − ac-full | auc | +0.0451 | 45/45 | 5.7e-14 | yes (+0.0364..+0.0539) |
| ac-full-saug − ac-full | cldice | +0.0815 | 45/45 | 5.7e-14 | yes (+0.0733..+0.0897) |
| no-router-saug − unet-saug | f1 | -0.0005 | 27/45 | 0.96 | no (-0.0029..+0.0020) |
| no-router-saug − unet-saug | auc | -0.0028 | 2/45 | 2.8e-13 | yes (-0.0047..-0.0010) |
| no-router-saug − unet-saug | cldice | +0.0015 | 27/45 | 0.08 | no (-0.0051..+0.0082) |
| ac-full-saug − unet-saug | f1 | -0.0003 | 21/45 | 0.79 | no (-0.0038..+0.0033) |
| ac-full-saug − unet-saug | auc | -0.0014 | 3/45 | 2.3e-09 | no (-0.0029..+0.0000) |
| ac-full-saug − unet-saug | cldice | +0.0005 | 27/45 | 0.36 | no (-0.0044..+0.0054) |
| ac-full-saug − no-router-saug | f1 | +0.0002 | 20/45 | 0.93 | no (-0.0010..+0.0013) |
| ac-full-saug − no-router-saug | auc | +0.0014 | 43/45 | 8.4e-10 | no (-0.0001..+0.0029) |
| ac-full-saug − no-router-saug | cldice | -0.0011 | 20/45 | 0.21 | no (-0.0040..+0.0018) |
