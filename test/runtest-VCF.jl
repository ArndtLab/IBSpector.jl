

@testset "Masking" begin
    m = VCF.Unmasked()
    @test VCF.isnotmasked(m, 1)
    @test VCF.transform_pos(m, 1) == 1

    v = [true, false, true, false]
    cmaskv = [1, 1, 2, 2]
    m2 = VCF.Masked{Int}(v, cmaskv)
    @test VCF.isnotmasked(m2, 1)
    @test !VCF.isnotmasked(m2, 2)
    @test VCF.isnotmasked(m2, 3)
    @test !VCF.isnotmasked(m2, 4)

    @test VCF.transform_pos(m2, 1) == 1
    @test VCF.transform_pos(m2, 3) == 2

    m3 = VCF.generate_mask(Int, v)
    @test m3.maskv == v
    @test m3.cmaskv == cmaskv
end


@testset "Read from pipe and with limit" begin
    using DataFrames

    vcfexamplesdir = "vcf-examples"

    file1 = joinpath(vcfexamplesdir, "1kGP_high_coverage_Illumina.chr1.filtered.SNV_INDEL_SV_phased_panel.first100000.vcf.gz")
    @test isfile(file1)

    # generate and read small file
    file2 = joinpath(vcfexamplesdir, "1kGP_high_coverage_Illumina.chr1.filtered.SNV_INDEL_SV_phased_panel.first100000.small.gz")
    if !isfile(file2)
        run(pipeline(file1, Cmd(`gzip -dc`, ignorestatus = true), ` head -2000`, `gzip -c`, file2))
    end
    @test isfile(file2)
    vcf_small = VCF.read(file2)

    # read large file with limit
    if true
        nr = nrow(vcf_small.df)
        vcf = VCF.read(file1, limit = nr)
        nr1 = DataFrames.nrow(vcf.df)
        @test nr1 <= nr # non-SNP might be filtered out, so we might get fewer rows than requested
        @test vcf.df[1:nr1, :] == vcf_small.df[1:nr1, :]
    end

    # read from pipe
    vcf2 = VCF.read(`gzip -dc $file2`)
    @test vcf_small.df == vcf2.df


    str = """
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
    chr21	5030578	21:5030578:C:T	C	T	.	.	AC=74	GT	0|0
    chr21	5030579	21:5030579:C:T	CA	T	.	.	AC=74	GT	0|0
    chr21	5030580	21:5030580:C:T	C	TC	.	.	AC=74	GT	0|0
    """

    vcf = VCF.read(IOBuffer(str))
    @test VCF.individuals(vcf) == ["HG00096"]
    @test vcf.df[1, VCF.colofindividual(vcf, "HG00096")] == "0|0"
    @test VCF.isphased(vcf, "HG00096")
    @test names(vcf.df)[1] == "CHROM"
    @test names(vcf.df)[end] == "HG00096"
    @test nrow(vcf.df) == 1
    @test_throws ErrorException VCF.colofindividual(vcf, "NonExisting")
end



@testset "read tskit generated vcf" begin

    vcfexamplesdir = "vcf-examples"
    file1 = joinpath(vcfexamplesdir, "test-tskit.vcf.gz")

    vcf = VCF.read(file1)

    indvs = VCF.individuals(vcf)
    @test length(indvs) == 10

    ibstl = VCF.IBStractlength(vcf, indvs[1])
    @test length(ibstl) > 0
    
    ibstl = mapreduce(vcat, VCF.individuals(vcf)) do indv
        @test VCF.isphased(vcf, indv)
        VCF.IBStractlength(vcf, indv) 
    end
    @test length(ibstl) == 353476

end


@testset "Test refs" begin
    str = """
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
    chr21	5	21:5:C:T	C	T	.	.	AC=74	GT	1|0
    """

    refs = Dict("chr21" => "AAAACTTTTT")
    vcf = VCF.read(IOBuffer(str), refs = refs)
    @test VCF.individuals(vcf) == ["HG00096"]

    ref = "AAAACTTTTT"
    vcf = VCF.read(IOBuffer(str), refs = ref)
    @test VCF.individuals(vcf) == ["HG00096"]



    refs = Dict(1 => ref) # wrong key type
    @test_throws ArgumentError VCF.read(IOBuffer(str), refs = refs)

    refs = Dict("chr1" => ref) # wrong chromosome name
    @test_throws ArgumentError VCF.read(IOBuffer(str), refs = refs)

    refs = Dict("chr21" => "TTTTTTT") # wrong reference allele
    @test_throws ArgumentError VCF.read(IOBuffer(str), refs = refs)


    str = """
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
    1	5	21:5:C:T	C	T	.	.	AC=74	GT	1|0
    """

    refs = Dict(1 => "AAAACTTTTT")
    vcf = VCF.read(IOBuffer(str), refs = refs)
    @test VCF.individuals(vcf) == ["HG00096"]

    ref = "AAAACTTTTT"
    vcf = VCF.read(IOBuffer(str), refs = ref)
    @test VCF.individuals(vcf) == ["HG00096"]

    refs = Dict(21 => ref) # wrong chromosome name
    @test_throws ArgumentError VCF.read(IOBuffer(str), refs = refs)

    refs = Dict(1 => "TTTTTTT") # wrong reference allele
    @test_throws ArgumentError VCF.read(IOBuffer(str), refs = refs)
end

@testset "Test IBStractlength" begin
    str = """
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
    chr21	5	21:5:C:T	C	T	.	.	AC=74	GT	1|0
    chr21	7	21:7:C:T	C	T	.	.	AC=74	GT	1|0
    chr21	10	21:10:C:T	C	T	.	.	AC=74	GT	1|0
    chr21	11	21:11:C:T	C	T	.	.	AC=74	GT	1|1
    """

    refs = Dict("chr21" => "CCCCCCCCCCCCCCCCCCC")
    vcf = VCF.read(IOBuffer(str), refs = refs)
    @test VCF.IBStractlength(vcf, "HG00096") == [2, 3]
end


@testset "Test IBStractlength with Segments" begin
    str = """
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
    chr21	5	21:5:C:T	C	T	.	.	AC=74	GT	1|0
    chr21	6	21:7:C:T	C	T	.	.	AC=74	GT	1|0
    chr21	10	21:10:C:T	C	A	.	.	AC=74	GT	1|0
    chr21	11	21:11:C:T	C	T	.	.	AC=74	GT	1|0
    chr21	12	21:11:C:T	C	T	.	.	AC=74	GT	1|0
    """

    ref ="CCCCCCCNCCCCCCCCCCCCCCCCCC"
    vcf = VCF.read(IOBuffer(str), refs = ref, segments = nothing)
    @test VCF.IBStractlength(vcf, "HG00096") == [1, 4, 1, 1]

    segments = VCF.generate_mask(map(i -> uppercase(i) == 'N', collect(ref)))
    vcf = VCF.read(IOBuffer(str), refs = ref, segments = segments)
    @test VCF.IBStractlength(vcf, "HG00096") == [1, 1, 1]

    vcf = VCF.read(IOBuffer(str), refs = ref, segments = :refnotN)
    @test VCF.IBStractlength(vcf, "HG00096") == [1, 1, 1]

    mask = 'P' ^ 10 * 'N' * 'P' ^ 10
    mask = VCF.generate_mask(map(==('P'), collect(mask)))
    vcf = VCF.read(IOBuffer(str), refs = ref, segments = :refnotN, masks = mask)
    @test VCF.IBStractlength(vcf, "HG00096") == [1, 1]
end





@testset "Double check with data we know already" begin
    using DataFrames

    vcfexamplesdir = "vcf-examples"

    file1 = joinpath(vcfexamplesdir, "1kGP_high_coverage_Illumina.chr1.filtered.SNV_INDEL_SV_phased_panel.first100000.vcf.gz")
    @test isfile(file1)

    seq = read(`gzip -dc $(joinpath(vcfexamplesdir, "seq.chr1-40M.txt.gz"))`, String)
    @test length(seq) == 40_000_000

    mask = read(`gzip -dc $(joinpath(vcfexamplesdir, "mask.chr1-40M.txt.gz"))`, String)
    @test length(mask) == 40_000_000

    ibs = read(`gzip -dc $(joinpath(vcfexamplesdir, "ilsHG00096.txt.gz"))`, String) |> split |> x -> parse.(Int, x)
    @test ibs[1:15] == [84, 57, 76, 1829, 1487, 279, 146, 3113, 542, 1895, 860, 193, 162, 1177, 4050]

    #generate mask from string
    gmask = VCF.generate_mask(map(==('P'), collect(mask)))

    vcf = VCF.read(file1, 
        refs = seq, 
        masks = gmask,
        segments = :refnotN,
        select = 1:10
        )
    @test nrow(vcf.df) > 1000
    @test "HG00096" in VCF.individuals(vcf)


    cibs = VCF.IBStractlength(vcf, "HG00096")
    ncibs = length(cibs)
    @test cibs == ibs[1:ncibs]

    indvs = VCF.individuals(vcf)
    @test length(indvs) == 1

    cibs = VCF.IBStractlength(vcf, [indvs[1], indvs[1]])
    @test cibs == vcat(ibs[1:ncibs], ibs[1:ncibs])

    cibs = VCF.IBStractlength(vcf)
    @test cibs == ibs[1:ncibs]
end

