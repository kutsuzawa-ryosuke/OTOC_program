using LinearAlgebra, Random, RandomMatrices
using Plots

function make_hamiltonian(Nq::Int) #横磁場イジング
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
    hamiltonian = XX + Zn
    return hamiltonian
end

function unitary(t)
    unitary = zeros(ComplexF64,(2^Nq,2^Nq))
    unitary = H_vec*diagm(exp.(-1.0im*H_val*t))*H_vec'
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

Igate = Matrix{ComplexF64}(I,2,2) #1-qubitの単位行列
Xgate = [0.0+0.0im 1.0+0.0im;1.0+0.0im 0.0+0.0im] #1-qubitのパウリX行列
Ygate = [0.0+0.0im 0.0-1.0im;0.0+1.0im 0.0+0.0im] ##1-qubitのパウリY行列
Zgate = [1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im] #1-qubitのパウリY行列

Nq = 10 #qubit数

β = 0 #最大10くらい
H = make_hamiltonian(Nq)
H_val, H_vec = eigen(H) 
ρ = exp(-1*β*H)/tr(exp(-β*H)) 
#=
A = make_pauli(1,Nq,"Z")
B = make_pauli(2,Nq,"X")

out = open("test.txt","a")
for t in 0:1:5
    println(out,"β=",β,',',"A=1",',',"B=",B_index)
    U(t) = unitary(t)
    OTOC(t) = tr(ρ*U(t)'*B'*U(t)*A'*U(t)'*B*U(t)*A)
    println(real(OTOC(t)))

    println(out,real(OTOC(t)),',',imag(OTOC(t)),',',abs(OTOC(t)))
    println(out,' ')
end
close(out)
=#
A = make_pauli(1,Nq,"Z")
B_list = [8,9,10]
β_list = [0,1,3,5]
for β in β_list
    ρ = exp(-1*β*H)/tr(exp(-β*H))
    for B_index in B_list
        out = open("Ising_no_RU.txt","a")
        B = make_pauli(B_index,Nq,"X")
        println("β=",β,',',"A=1",',',"B=",B_index)
        println(out,"β=",β,',',"A=1",',',"B=",B_index)
        for t in 150:1:200
            U(t) = unitary(t)
            OTOC(t) = tr(ρ*U(t)'*B'*U(t)*A'*U(t)'*B*U(t)*A)
            println(t)
            println(out,OTOC(t))
        end
        println(out,' ')
        close(out)
    end
end