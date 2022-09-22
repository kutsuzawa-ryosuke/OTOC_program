using LinearAlgebra, Random, RandomMatrices
using Plots

Igate = Matrix{ComplexF64}(I,2,2)                  #1-qubitの単位行列
Xgate = [0.0+0.0im 1.0+0.0im;1.0+0.0im 0.0+0.0im]  #1-qubitのパウリX行列
Ygate = [0.0+0.0im 0.0-1.0im;0.0+1.0im 0.0+0.0im]  #1-qubitのパウリY行列
Zgate = [1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im] #1-qubitのパウリY行列

function RandomUnitaryMatrix(Nq,theta1,theta2)
    I_gate = Matrix{ComplexF64}(I,2^(Nq-2),2^(Nq-2))
    X1 = [cos(theta1) -im*sin(theta1)
        -im*sin(theta1) cos(theta1)]
    X2 = [cos(theta2) -im*sin(theta2)
        -im*sin(theta2) cos(theta2)]
    RU = kron(X1,X2)
    RU = kron(RU,I_gate)
    return RU
end

function make_hamiltonian(Nq::Int)
    XX = zeros(ComplexF64,(2^Nq,2^Nq))
    Zn = zeros(ComplexF64,(2^Nq,2^Nq))
    hamiX = zeros(ComplexF64,(2,2))
    hamiZ = zeros(ComplexF64,(2,2))

    for k in 1:Nq-1
        for l in 1:Nq
            if k == l
                if l == 1
                    hamiX = Xgate
                else
                    hamiX = kron(hamiX,Xgate)
                end
            elseif k+1 == l
                hamiX = kron(hamiX,Xgate)
            else
                if l == 1
                    hamiX = Igate
                else
                    hamiX = kron(hamiX,Igate)
                end      
            end    
        end
        XX = XX + hamiX
    end

    for q in 1:Nq
        for r in 1:Nq
            if q == r
                if r == 1
                    hamiZ = Zgate
                else
                    hamiZ = kron(hamiZ,Zgate)
                end
            else
                if r == 1
                    hamiZ = Igate
                else
                    hamiZ = kron(hamiZ,Igate)
                    
                end 
            end
        end
        Zn = Zn + hamiZ
    end
    hamiltonian = (-1)*(XX + Zn)
    return hamiltonian
end

function make_unitary_pool(Nq::Int,S,θ,ϕ)
    unitary_pool = zeros(ComplexF64,2^Nq,2^Nq,N)

    for i in 1:length(S)
        hamiltonian_unitary = H_vec*diagm(exp.(-1.0im*H_val*S[i]))*H_vec'
        #hamiltonian_unitary = exp(-1.0im*H*S[i])
        if i == 1
            unitary_pool[:,:,i] = hamiltonian_unitary
        else
            RU = RandomUnitaryMatrix(Nq,θ[i-1],ϕ[i-1])
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
A = make_pauli(1,Nq,"Z")
#B = make_pauli(5,Nq,"X")

TimeArray = zeros(Float64,N,N)
θ = zeros(Float64,N,N)
ϕ = zeros(Float64,N,N)
for i in 1:1:N
    for j in 1:N
        s = rand()*10
        θ[i,j] = rand()*2*pi
        ϕ[i,j] = rand()*2*pi
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
B_list = [2,3,4,5,6,7,8,9,10]
function main()
    for β in β_list
        ρ = exp(-1*β*H)/tr(exp(-β*H))
        for B_index in B_list
            out = open("Ising_β=$β(1,$B_index).txt","a")
            
            B = make_pauli(B_index,Nq,"X")
            println("β=",β,',',"A=1",',',"B=",B_index)
            result = zeros(ComplexF64,N,T+1)
            result_ave = zeros(ComplexF64,T+1)

            for i in 1:1:N
                println(TimeArray[i,:])
                println(out,TimeArray[i,:])
                unitary_pool = make_unitary_pool(Nq,TimeArray[i,:],θ[i,:],ϕ[i,:])
                
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
T_Old = 130 #前回のTimeMax
TimeMaxArray = [
104.20421349396803
121.20941621467182
81.73271877431576
87.14351189740607
81.97278070863186
112.05697968085103
98.61386649231636
101.3945583148395
98.44048434107557
115.02085150358047
111.1777661947926
126.32972566920571
129.28844631327559
121.67654505822183
64.45071408281214
86.33171855741386
96.82166397217668
98.54475467934446
93.69112821534962
91.29010376282142
]
#RUを追加する時のRUが入る時間
TimeAddArray = zeros(Float64,N,ADD)
θ = zeros(Float64,N,ADD)
ϕ = zeros(Float64,N,ADD)
for i in 1:1:N
    for j in 1:ADD
        s = rand()*10
        θ[i,j] = rand()*2*pi
        ϕ[i,j] = rand()*2*pi
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
            out = open("Ising_β=$β(1,$B_index)_Add.txt","a")
            
            B = make_pauli(B_index,Nq,"X")
            println("β=",β,',',"A=1",',',"B=",B_index)
            result = zeros(ComplexF64,N,T_New-T_Old)
            result_ave = zeros(ComplexF64,T_New-T_Old)

            for i in 1:1:N
                println(TimeAddArray[i,:])
                println(out,TimeAddArray[i,:])
                unitary_pool = make_unitary_pool(Nq,TimeAddArray[i,:],θ[i,:],ϕ[i,:])
                k = 1
                for t in T_Old+1:1:T_New
                    println(t)
                    U = chose_unitary(unitary_pool,TimeAddArray[i,:],t)
                    OTOC = tr(ρ*U'*B'*U*A'*U'*B*U*A)
                    #println(out,OTOC)
                    #println(OTOC)
                    result[i,k] = OTOC
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





for i in 1:1:10
        
    S = zeros(Float64,N)
    θ = zeros(Float64,N)
    ϕ = zeros(Float64,N)
    T = zeros(Float64,N)
    for i in 1:N
        S[i] = rand()*10
        θ[i] = rand()*2*pi
        ϕ[i] = rand()*2*pi
        #println(T)
        if i == 1
            T[i] = S[i]
        else
            T[i] = T[i-1]+S[i]
        end
    end
    println(S)
    println(T)
    unitary_pool = make_unitary_pool(Nq,S,θ,ϕ)

    β_list = [0,1,3,5]
    B_list = [8,9,10]
    for β in β_list
        ρ = exp(-1*β*H)/tr(exp(-β*H))
        for B_index in B_list
            out = open("Ising_β=$β(1,$B_index).txt","a")
            println(out,S)
            println(out,T)
            
            B = make_pauli(B_index,Nq,"X")
            println("β=",β,',',"A=1",',',"B=",B_index)
            for t in 0:1:150
                U(t) = chose_unitary(unitary_pool,S,t)
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