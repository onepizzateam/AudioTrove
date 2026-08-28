class Audiotrove < Formula
  desc "CPU-first audio curation and Piper training pipeline"
  homepage "https://github.com/onepizzateam/AudioTrove"
  url "https://github.com/onepizzateam/AudioTrove/archive/refs/tags/v0.1.1.tar.gz"
  version "0.1.1"
  license "MIT"

  depends_on "python@3.11"

  def install
    system "#{Formula[\"python@3.11\"].opt_bin}/pip", "install", *std_pip_args, "."
  end

  test do
    system bin/"audiotrove", "doctor"
  end
end
