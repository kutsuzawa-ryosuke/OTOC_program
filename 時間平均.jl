F = [
1
-0.50009787
0.742520793
0.895026506
0.945509148
0.97671
0.989578377
0.991406478
0.990202582
0.957507778
0.362079649
-0.157285694
0.180408451
0.268037655
0.296836047
0.378317953
0.440760151
0.439283249
0.44217238
0.508988371
0.350052454
0.214907846
-0.019828861
0.229593431
0.226456191
0.419285476
0.398783724
0.284908635
0.220583595
0.293500854
0.411952722
0.393468
0.198607702
0.179794912
0.184271143
0.246109723
0.397017283
0.297950095
0.149937567
0.207264992
0.330526363
0.414896176
0.313471948
0.100006866
0.222403315
0.313285964
0.26718092
0.17523584
0.162459264
0.300655453
0.388746128
0.264158183
0.237497614
0.223094437
0.153468486
0.155424641
0.173227105
0.290330142
0.335761574
0.324740134
0.276322461
0.194732243
0.152375013
0.137826047
0.174940525
0.251480567
0.247920738
0.14933589
0.153488373
0.232627539
0.291231307
0.244111667
0.204559059
0.190854875
0.144925309
0.127000355
0.321716985
0.314616277
0.254142391
0.238925425
0.276669453
0.305862027
0.261599155
0.237144944
0.165350536
0.028120339
0.0547847
0.115071062
0.281627141
0.25837138
0.147647586
0.157806458
0.242755644
0.265423931
0.277689117
0.241230036
0.177822548
0.211958709
0.213721705
0.213741985
0.209987347
0.157103718
0.134523002
0.153547202
0.198066272
0.259586878
0.155064681
0.106184407
0.166089336
0.191939013
0.170840305
0.208766995
0.261027565
0.196654177
0.140419695
0.143190226
0.217790202
0.160031572
0.131906532
0.173748402
0.176431639
0.166513329
0.193243613
0.189412225
0.182360308
0.163022775
0.154982857
0.204728787
0.144221034
0.117666097
0.144444794]

K = 5
L = length(F)-K+1



#f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9),
#g(i)={ f(i)+f(i+1)+....+f(i+k-1) }/k

function G(i)
    g = 0
    for k in 0:1:K-1
        g = g + F[i+k]
    end
    g = g/K
    return g
end

for i in 1:1:L
    println(G(i))
end