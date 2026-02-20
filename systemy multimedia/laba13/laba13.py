class StorageMedia:
    def __init__(self, name, capacity, weight, height, volume, price):
        self.name = name
        self.capacity = capacity
        self.weight = weight
        self.height = height
        self.volume = volume
        self.price = price


class InformationCalculator:
    def __init__(self):
        self.utf16BytesPerChar = 2
        self.storageMedia = self.initializeStorageMedia()

    def initializeStorageMedia(self):
        media = {}

        media["a4Paper"] = StorageMedia(
            name="A4 Paper (12pt font, 2.5cm margins)",
            capacity=3000 * self.utf16BytesPerChar,
            weight=0.005,
            height=0.0001,
            volume=0.21 * 0.297 * 0.0001,
            price=0.05
        )

        media["hdd4tb"] = StorageMedia(
            name="FireCuda 530R SSD 4 TB (https://www.seagate.com/as/en/products/gaming-drives/pc-gaming/firecuda-530r-ssd/)",
            capacity=4 * 1024**4,
            weight=0.70,
            height=0.046,
            volume=0.14699 * 0.1016 * 0.026,
            price=419.12
        )

        media["ssd128gb"] = StorageMedia(
            name="Kingston A400 SSD 128GB (https://www.kingston.com/en/ssd/a400-solid-state-drive)",
            capacity=128 * 1024**3,
            weight=0.041,
            height=0.007,
            volume=0.1 * 0.07 * 0.007,
            price=32.0
        )

        media["cd"] = StorageMedia(
            name="Verbatim CD-R 700MB (https://www.verbatim-europe.com/en/prod/cd-r-52x-700mb-80min-50pk-spindle-43343/)",
            capacity=700 * 1024**2,
            weight=0.015,
            height=0.0012,
            volume=3.1416 * (0.06 ** 2) * 0.0012,
            price=0.8
        )

        media["dvd"] = StorageMedia(
            name="Verbatim DVD-R 4.7GB (https://www.verbatim-europe.com/en/prod/dvd-r-16x-4-7gb-50pk-spindle-43549/)",
            capacity=int(4.7 * 1024**3),
            weight=0.016,
            height=0.0012,
            volume=3.1416 * (0.06 ** 2) * 0.0012,
            price=1.2
        )

        media["microSd"] = StorageMedia(
            name="SanDisk Ultra microSDXC 64GB (https://shop.sandisk.com/pl-pl/products/memory-cards/microsd-cards/sandisk-ultra-lite-uhs-i-microsd-without-adapter?sku=SDSQUNR-064G-GN3MN)",
            capacity=64 * 1024**3,
            weight=0.00025,
            height=0.001,
            volume=0.015 * 0.011 * 0.001,
            price=10.0
        )

        return media


    def calculateBookSize(self, pages=300, charsPerLine=60, linesPerPage=25):
        totalChars = pages * linesPerPage * charsPerLine
        totalBytes = totalChars * self.utf16BytesPerChar
        return totalBytes, totalChars

    def calculateEncyclopediaBritannica(self):
        volumes = 32
        pagesPerVolume = 1000
        charsPerPage = 3000
        totalChars = volumes * pagesPerVolume * charsPerPage
        totalBytes = totalChars * self.utf16BytesPerChar
        return totalBytes, totalChars

    def calculateWikipedia(self, articles=1500000, avgCharsPerArticle=5000):
        totalChars = articles * avgCharsPerArticle
        totalBytes = totalChars * self.utf16BytesPerChar
        return totalBytes, totalChars

    def calculateUncompressedPhoto(self, megapixels=108, bitsPerPixel=24):
        totalPixels = megapixels * 1000000
        totalBits = totalPixels * bitsPerPixel
        totalBytes = totalBits // 8
        return totalBytes

    def calculateUncompressedVideo(self, width, height, fps, durationMinutes):
        pixels = width * height
        frames = fps * durationMinutes * 60
        bitsPerPixel = 24
        totalBits = pixels * frames * bitsPerPixel
        totalBytes = totalBits // 8
        return totalBytes

    def calculateYouTubeHourlyData(self):
        hoursUploadedPerHour = 500
        avgQuality = "720p"
        avgBitrate = 2500000
        avgDurationMinutes = 10
        avgFileSizeBytes = (avgBitrate * avgDurationMinutes * 60) // 8
        totalBytesPerHour = hoursUploadedPerHour * 60 * avgFileSizeBytes
        return totalBytesPerHour

    def calculateStorageRequirements(self, dataSize, mediaType):
        media = self.storageMedia[mediaType]
        unitsNeeded = (dataSize + media.capacity - 1) // media.capacity

        totalCost = unitsNeeded * media.price
        totalWeight = unitsNeeded * media.weight
        totalHeight = unitsNeeded * media.height
        totalVolume = unitsNeeded * media.volume

        return {
            "unitsNeeded": unitsNeeded,
            "totalCost": totalCost,
            "totalWeight": totalWeight,
            "totalHeight": totalHeight,
            "totalVolume": totalVolume,
        }

    def formatBytes(self, bytes):
        if bytes < 1024:
            return f"{bytes} B"
        elif bytes < 1024**2:
            return f"{bytes/1024:.2f} KB"
        elif bytes < 1024**3:
            return f"{bytes/(1024**2):.2f} MB"
        elif bytes < 1024**4:
            return f"{bytes/(1024**3):.2f} GB"
        else:
            return f"{bytes/(1024**4):.2f} TB"

    def runCalculations(self):

        print("PART 1: REFERENCE STORAGE MEDIA")
        print("-" * 50)
        for key, media in self.storageMedia.items():
            print(f"{media.name}:")
            print(f"  Capacity: {self.formatBytes(media.capacity)}")
            print(f"  Weight: {media.weight} kg")
            print(f"  Height: {media.height*1000:.2f} mm")
            print(f"  Volume: {media.volume*1000000:.2f} cm³")
            print(f"  Price: ${media.price:.2f}")
            print()

        print("\nPART 2: DATA SIZE ESTIMATIONS")
        print("-" * 50)

        bookBytes, bookChars = self.calculateBookSize()
        print(
            f"Book (300 pages): {self.formatBytes(bookBytes)} ({bookChars:,} characters)"
        )

        encBytes, encChars = self.calculateEncyclopediaBritannica()
        print(
            f"Encyclopedia Britannica: {self.formatBytes(encBytes)} ({encChars:,} characters)"
        )

        wikiBytes, wikiChars = self.calculateWikipedia()
        print(f"Wikipedia: {self.formatBytes(wikiBytes)} ({wikiChars:,} characters)")

        photoBytes = self.calculateUncompressedPhoto()
        print(f"108MP Uncompressed Photo: {self.formatBytes(photoBytes)}")

        vhsBytes = self.calculateUncompressedVideo(720, 576, 25, 90)
        print(f"VHS (PAL) 1.5h Uncompressed: {self.formatBytes(vhsBytes)}")

        fullHdBytes = self.calculateUncompressedVideo(1920, 1080, 30, 90)
        print(f"FullHD 1.5h Uncompressed: {self.formatBytes(fullHdBytes)}")

        fourKBytes = self.calculateUncompressedVideo(3840, 2160, 60, 90)
        print(f"4K 1.5h Uncompressed: {self.formatBytes(fourKBytes)}")

        youtubeBytes = self.calculateYouTubeHourlyData()
        print(f"YouTube Data per Hour: {self.formatBytes(youtubeBytes)}")

        print("\nPART 3: STORAGE REQUIREMENTS ANALYSIS")
        print("-" * 50)

        datasets = {
            "Book": bookBytes,
            "Encyclopedia Britannica": encBytes,
            "Wikipedia": wikiBytes,
            "108MP Photo": photoBytes,
            "VHS 1.5h": vhsBytes,
            "FullHD 1.5h": fullHdBytes,
            "4K 1.5h": fourKBytes,
            "YouTube/hour": youtubeBytes,
        }

        for dataName, dataSize in datasets.items():
            print(f"\n{dataName} ({self.formatBytes(dataSize)}):")
            print("  Storage Medium Requirements:")

            for mediaKey, media in self.storageMedia.items():
                req = self.calculateStorageRequirements(dataSize, mediaKey)
                if req["unitsNeeded"] > 0:
                    print(f"    {media.name}:")
                    print(f"      Units needed: {req['unitsNeeded']:,}")
                    print(f"      Total cost: ${req['totalCost']:,.2f}")
                    print(f"      Total weight: {req['totalWeight']:.3f} kg")
                    if req["totalHeight"] > 1:
                        print(f"      Tower height: {req['totalHeight']:.2f} m")
                    else:
                        print(f"      Tower height: {req['totalHeight']*1000:.2f} mm")
                    print(f"      Total volume: {req['totalVolume']*1000000:.2f} cm³")


def main():
    calculator = InformationCalculator()
    calculator.runCalculations()


if __name__ == "__main__":
    main()

