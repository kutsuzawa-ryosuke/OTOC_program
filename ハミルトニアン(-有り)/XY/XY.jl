using LinearAlgebra, Random, RandomMatrices
using Plots

Igate = Matrix{ComplexF64}(I,2,2)                  #1-qubitの単位行列
Xgate = [0.0+0.0im 1.0+0.0im;1.0+0.0im 0.0+0.0im]  #1-qubitのパウリX行列
Ygate = [0.0+0.0im 0.0-1.0im;0.0+1.0im 0.0+0.0im]  #1-qubitのパウリY行列
Zgate = [1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im] #1-qubitのパウリY行列

function RandomUnitaryMatrix(Nq::Int,Dim::Int)
    I_gate = Matrix{ComplexF64}(I,2^(Nq-2),2^(Nq-2))
    x = (randn(Dim,Dim) + randn(Dim,Dim)*im)/ sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    RU = f.Q * diagRm
    RU = kron(RU,I_gate)
    return RU
end

function make_hamiltonian(Nq::Int)#XYモデル
    XX = zeros(ComplexF64,(2^Nq,2^Nq))
    YY = zeros(ComplexF64,(2^Nq,2^Nq))
    hamiX = zeros(ComplexF64,(2^Nq,2^Nq))
    hamiY = zeros(ComplexF64,(2^Nq,2^Nq))

    for k in 1:Nq-1
        for l in 1:Nq
            if k == l
                if l == 1
                    hamiX = Xgate
                    hamiY = Ygate
                else
                    hamiX = kron(hamiX,Xgate)
                    hamiY = kron(hamiY,Ygate)
                end

            elseif k+1 == l
                hamiX = kron(hamiX,Xgate)
                hamiY = kron(hamiY,Ygate)

            else
                if l == 1
                    hamiX = Igate
                    hamiY = Igate
                else
                    hamiX = kron(hamiX,Igate)
                    hamiY = kron(hamiY,Igate)
                    
                end      
            end    
        end
        XX = XX + 0.5*hamiX
        YY = YY + 0.5*hamiY
        
    end
    hamiltonian = (-1)*(XX + YY)
    return hamiltonian
end

function make_unitary_pool(Nq::Int,S)
    unitary_pool = zeros(ComplexF64,2^Nq,2^Nq,N)

    for i in 1:length(S)#N
        hamiltonian_unitary = H_vec*diagm(exp.(-1.0im*H_val*S[i]))*H_vec'
        #hamiltonian_unitary = exp(-1.0im*H*S[i])
        if i == 1
            unitary_pool[:,:,i] = hamiltonian_unitary
        else
            RU = RandomUnitaryMatrix(Nq,4)
            unitary_pool[:,:,i] = unitary_pool[:,:,i-1]*RU*hamiltonian_unitary
        end 
    end
    return unitary_pool
end

function chose_unitary(unitary_pool,S,t)
    s = 0
    i = 1
    unitary = zeros(ComplexF64,(2^Nq,2^Nq))
    while i <= length(S)
        if t < S[1]
            unitary = H_vec*diagm(exp.(-1.0im*H_val*t))*H_vec'
            return unitary
            break
        
        else
            if s <= t < s+S[i]
                unitary = unitary_pool[:,:,i-1]*(H_vec*diagm(exp.(-1.0im*H_val*(t-s)))*H_vec')
                return unitary 
                break
            end
        end
        s += S[i]
        i += 1
    end
    unitary = unitary_pool[:,:,N]*(H_vec*diagm(exp.(-1.0im*H_val*(t-s)))*H_vec')
    return unitary
end

function make_pauli(index::Int, Nq::Int, pauli_name)
    pauli = zeros(ComplexF64,(2^Nq,2^Nq))
    for i in 1:Nq
        if i == 1 #一番最初のループはおおもとになる行列の宣言をする(pauliを作る)
            if index == 1 #1st qubitがXかYのとき
                if pauli_name == "X"
                    pauli = Xgate
                elseif pauli_name == "Y"
                    pauli = Ygate
                elseif pauli_name == "Z"
                    pauli = Zgate
                else
                    println("ERROR: undefined!")
                    return False
                end
            else
                pauli = Igate
            end
        else #2ループ目以降は変数pauliがあるはずなので、pauli \tensor (X or Z or I)を計算してpauliを上書き代入
            if i == index #XかYを代入するタイミングのとき
                if pauli_name == "X"
                    pauli = kron(pauli, Xgate)
                elseif pauli_name == "Y"
                    pauli = kron(pauli, Ygate)
                elseif pauli_name == "Z"
                    pauli = kron(pauli, Zgate)
                end
            else
                pauli = kron(pauli, Igate)
            end
        end
    end
    return pauli
end

Nq = 10 #qubit数
N = 20  #乱数取る回数
#β = 0  #最大10くらい
H = make_hamiltonian(Nq)
H_val, H_vec = eigen(H) 
#ρ = exp(-1*β*H)/tr(exp(-β*H)) 
A = make_pauli(4,Nq,"X")
#B = make_pauli(5,Nq,"X")


#RUの入る時間を決定
TimeArray = zeros(Float64,N,N)
for i in 1:1:N
    for j in 1:N
        s = rand()*10
        if j == 1
            TimeArray[i,j] = s
        else
            TimeArray[i,j] = TimeArray[i,j-1]+s
        end
    end
end
TimeMax = maximum(TimeArray)
T = Int(round(TimeMax) + 1)
TimeArray[:,:]
size(TimeArray)
out = open("time.txt","a")
println(out,TimeArray)
println(out,"")
for i in 1:1:N
    println(out,TimeArray[i,:])
end
println(out,"TimeMax")
for i in 1:1:N
    println(out,maximum(TimeArray[i,:]))
end
close(out)


β_list = [0,1,3,5]
B_list = [5,8,10]
function main()
    for β in β_list
        ρ = exp(-1*β*H)/tr(exp(-β*H))
        for B_index in B_list
            out = open("XY_β=$β(4,$B_index).txt","a")
            
            B = make_pauli(B_index,Nq,"Y")
            println("β=",β,',',"A=4",',',"B=",B_index)
            result = zeros(ComplexF64,N,T+1)
            result_ave = zeros(ComplexF64,T+1)

            for i in 1:1:N #平均を取る回数
                println(TimeArray[i,:])
                println(out,TimeArray[i,:])
                unitary_pool = make_unitary_pool(Nq,TimeArray[i,:])
                
                for t in 0:1:T
                    println(t)
                    U = chose_unitary(unitary_pool,TimeArray[i,:],t)
                    OTOC = tr(ρ*U'*B'*U*A'*U'*B*U*A)
                    #println(out,OTOC)
                    #println(OTOC)
                    result[i,t+1] = OTOC
                end
                
                for j in 1:1:T+1
                    println(out,result[i,j])
                    result_ave[j] += result[i,j]
                end
                println(out,"")
            end
            println(out,"average")
            result_ave = result_ave ./ N
            for j in 1:1:T+1
                println(out,result_ave[j])
            end
            println(out,"")
            println(out, "最後にRUが入ったタイミング: ",TimeMax)
            close(out)
        end
    end
end

main()

ADD = 10 #増やすRUの数
T_Old = 118 #前回のTimeMax
TimeMaxArray = [
100.8981413707141
70.87872933146006
98.19790464432332
103.71869666045494
100.67130492614498
117.36779869002751
104.61305879000378
72.51120492702188
91.75526515712038
87.447574736012
78.22967381333072
115.32590528050702
101.97008522090645
111.63273318755077
94.75873500751084
110.71975060260321
92.35518666373613
97.3336147327156
83.69219991240709
110.44050564030688]

#RUを追加する時のRUが入る時間
TimeAddArray = zeros(Float64,N,ADD)
for i in 1:1:N
    for j in 1:ADD
        s = rand()*10
        if j == 1
            TimeAddArray[i,j] = TimeMaxArray[i]+s
        else
            TimeAddArray[i,j] = TimeAddArray[i,j-1]+s
        end
        
    end
end
TimeAddArray
TimeAddMax = maximum(TimeAddArray)
T_New = Int(round(TimeAddMax) + 1)
size(TimeAddArray)
out = open("time_Add.txt","a")
print(out,TimeAddArray)
println(out,"")
for i in 1:1:N
    println(out,TimeAddArray[i,:])
end
println(out,"")
for i in 1:1:N
    println(out,maximum(TimeAddArray[i,:]))
end
close(out)

function main_Add()
    for β in β_list
        ρ = exp(-1*β*H)/tr(exp(-β*H))
        for B_index in B_list
            out = open("XY_β=$β(4,$B_index)_Add.txt","a")
            
            B = make_pauli(B_index,Nq,"Y")
            println("β=",β,',',"A=4",',',"B=",B_index)
            result = zeros(ComplexF64,N,T_New-T_Old)
            result_ave = zeros(ComplexF64,T_New-T_Old)

            for i in 1:1:N
                println(TimeAddArray[i,:])
                println(out,TimeAddArray[i,:])
                unitary_pool = make_unitary_pool(Nq,TimeAddArray[i,:])
                k = 1
                for t in T_Old+1:1:T_New
                    println(t)
                    U = chose_unitary(unitary_pool,TimeAddArray[i,:],t)
                    OTOC = tr(ρ*U'*B'*U*A'*U'*B*U*A)
                    #println(out,OTOC)
                    #println(OTOC)
                    result[i,k] = OTOC
                    k = k+1
                end
                
                for j in 1:1:T_New-T_Old
                    println(out,result[i,j])
                    result_ave[j] += result[i,j]
                end
                println(out,"")
            end
            println(out,"average")
            result_ave = result_ave ./ N
            for j in 1:1:T_New-T_Old
                println(out,result_ave[j])
            end
            println(out,"")
            println(out, "最後にRUが入ったタイミング: ",TimeAddMax)
            close(out)
        end
    end
end

main_Add()



for i in 1:1:N
    unitary_pool = make_unitary_pool(Nq,TimeArray[i,:])

    β_list = [0,1]
    B_list = [4,5]
    for β in β_list
        ρ = exp(-1*β*H)/tr(exp(-β*H))
        for B_index in B_list
            out = open("XY_β=$β(4,$B_index).txt","a")
            println(out,TimeArray[i,:])
            println(out,"")
            
            B = make_pauli(B_index,Nq,"Y")
            println("β=",β,',',"A=4",',',"B=",B_index)
            for t in 0:1:TimeMax+5
                U(t) = chose_unitary(unitary_pool,TimeArray[i,:],t)
                OTOC(t) = tr(ρ*U(t)'*B'*U(t)*A'*U(t)'*B*U(t)*A)
                #println(real(OTOC(t)))
                println(t)
                println(out,OTOC(t))
            end
            println(out,"")
            close(out)
        end
    end
end