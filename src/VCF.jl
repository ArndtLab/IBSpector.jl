module VCF

using CSV
using DataFrames
using DataFramesMeta



abstract type AbstractMask end

struct Unmasked <: AbstractMask end

isnotmasked(m::Unmasked, p) = true
transform_pos(m::Unmasked, p) = p


struct Masked{T} <: AbstractMask
    maskv::Vector{Bool}
    cmaskv::Vector{T}
end

isnotmasked(m::Masked{T}, p) where T = m.maskv[p]
transform_pos(m::Masked{T}, p) where T = m.cmaskv[p]

generate_mask(T, v::Vector{Bool}) = 
    Masked{T}(v, accumulate(+, v, init = zero(T)))

generate_mask(v::Vector{Bool}) = generate_mask(Int, v)



struct VCFdata 
    df::DataFrame
end


function read(file; 
    refs = nothing,
    masks = nothing,
    segments = nothing,
    kwargs...)

    df = CSV.read(file, DataFrame; 
        comment = "##", 
        buffer_in_memory = true, 
        ntasks = Threads.nthreads(),
        kwargs...
        );
    rename!(df, names(df)[1] => replace(String(names(df)[1]), "#" => ""))
    # only keep SNVs
    subset!(df, :REF => ByRow(x -> length(x) == 1), :ALT => ByRow(x -> length(x) == 1)) 
        
    refs = dictify(df, refs)
    check_keyypes(df, refs)

    isnothing(refs) || checkrefs(VCFdata(df), refs)
        
    if isnothing(segments)
        insertcols!(df, :REF, :segment => 1)
    else
        segments = dictify(df, segments)
        check_keyypes(df, segments)
        segments = map(keys(segments), values(segments)) do k, v
            if v == :refnotN
                v = generate_mask(map(i -> uppercase(i) == 'N', collect(refs[k])))
            end
            (k, v)
        end |> Dict
        insertcols!(df, :REF, :segment => 1)
        @rtransform! df :segment = transform_pos(segments[:CHROM], :POS)
    end
    
    if isnothing(masks)
        rename!(df, :POS => :MASKED_POS)
    else
        masks = dictify(df, masks)
        check_keyypes(df, masks)
        @rsubset! df isnotmasked(masks[:CHROM], :POS)
        insertcols!(df, :POS, :MASKED_POS => 1)
        @rtransform! df :MASKED_POS = transform_pos(masks[:CHROM], :POS)
    end
    
    vcf = VCFdata(df)
    return vcf
end


function dictify(df, s)
    isnothing(s) && return nothing
    if !(s isa Dict)
        return Dict(map(unique(df[!, 1])) do chrom
            (chrom, s)
        end)
    end
    return s
end

function check_keyypes(df, dict)
    if isnothing(dict)
        return
    end
    if !(
        (keytype(dict) <: AbstractString) && (eltype(df[!, 1]) <: AbstractString) ||
        (keytype(dict) <: Integer) && (eltype(df[!, 1]) <: Integer)
        )
        throw(ArgumentError("dict has not compatible key type with chromosome names in VCF."))
    end
    for chr in unique(df[!, 1])
        haskey(dict, chr) || throw(ArgumentError("chromosome $chr in VCF not found in dict."))
    end
end


function checkrefs(vcf::VCFdata, refs)
    for r in eachrow(vcf.df)
        chrom = r[1]
        pos = r[2]
        ref = first(r[:REF])
        refx = refs[chrom][pos]
        if uppercase(ref) != uppercase(refx)
            throw(ArgumentError("Reference allele mismatch at $chrom:$pos: VCF has $ref, but refs has $refx"))
        end
    end
end



function individuals(vcf::VCFdata)
    indvs = findfirst(==("FORMAT"), names(vcf.df))+1: length(names(vcf.df))
    return names(vcf.df)[indvs]
end

function colofindividual(vcf::VCFdata, indv::AbstractString)
    c = findfirst(==(indv), names(vcf.df))
    if c === nothing
        error("Individual $indv not found in VCF.")
    end
    return c
end


function isphased(vcf::VCFdata, indv::AbstractString)
    c = colofindividual(vcf, indv)
    phased = ["1|0", "0|1", "1|1", "0|0", "0/0", "1/1"]
    return all(x -> x in phased, vcf.df[!, c])
end

function ibstl(v)
    length(v) <= 1 && return empty(v)
    v[2:end] .- v[1:end-1] 
end



function IBStractlength(vcf::VCFdata, indv::AbstractString)
    c1 = colofindividual(vcf, indv)

    tmp = @chain vcf.df begin
        subset(c1 =>  c -> c .== "1|0" .|| c .== "0|1")
        groupby([:CHROM, :segment])
        @combine :ibstl = Ref(ibstl(:MASKED_POS))
    end 
    filter(!=(0), reduce(vcat, tmp.ibstl, init = Int[]))
end

function IBStractlength(vcf::VCFdata, indvs::Vector{T} = individuals(vcf)) where T <: AbstractString
    mapreduce(indv -> IBStractlength(vcf, indv), vcat, indvs, init = Int[])
end


end # module